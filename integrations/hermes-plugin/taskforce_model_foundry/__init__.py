"""Hermes directory plugin that mounts Taskforce Model Foundry tools.

Hermes discovers a directory plugin by ``plugin.yaml`` + a ``register(ctx)``
in this module, and tools reach the model only through
``ctx.register_tool()`` -> ``tools.registry.register()``. The Foundry action
table lives in Node (``src/integrations/hermes.mjs``), so this module is a
thin mount: it asks the Node bridge for the adapter's own tool specs and
registers one handler per spec. Nothing is restated here, so an action added
on the Node side appears on Hermes without touching this file.

Resolution of the Foundry checkout, in order:
  1. ``TMF_FOUNDRY_ROOT``
  2. ``<plugin dir>/foundry`` (a copy or symlink placed next to the plugin)
  3. ``~/.hermes/taskforce-model-foundry``

Resolution of ``node``, in order: ``TMF_NODE``, then ``PATH``, then the usual
per-user/package locations (see ``_NODE_FALLBACKS``). The fallbacks exist
because a gateway started as a service does not inherit the login shell's
PATH, and without them a per-user node install makes the mount skip silently.

Registration never raises: a missing checkout or a broken bridge is reported
as a skipped mount, because a plugin that throws in ``register`` takes the
host's whole plugin load path down with it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
TOOLSET = "taskforce_model_foundry"
_BRIDGE = Path(__file__).resolve().parent / "bridge.mjs"
_TIMEOUT_SPECS = 30
_TIMEOUT_CALL = 1800


def _foundry_root() -> Path | None:
    candidates = []
    env = os.environ.get("TMF_FOUNDRY_ROOT")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(_BRIDGE.parent / "foundry")
    candidates.append(Path.home() / ".hermes" / "taskforce-model-foundry")
    for candidate in candidates:
        if (candidate / "src" / "integrations" / "hermes.mjs").is_file():
            return candidate.resolve()
    return None


# A gateway started as a systemd service inherits a cropped PATH, not the
# login shell's, so a per-user node (nvm, ~/.local/bin) is invisible to the
# plugin even though the operator can run `node` interactively. Looking in the
# usual per-user and package locations turns that silent skipped mount into a
# working one; TMF_NODE stays available for anything unusual.
_NODE_FALLBACKS = (
    "~/.local/bin/node",
    "/usr/local/bin/node",
    "/opt/homebrew/bin/node",
    "/usr/bin/node",
)


def _node() -> str | None:
    override = os.environ.get("TMF_NODE")
    if override:
        candidate = Path(override).expanduser()
        return str(candidate) if os.access(candidate, os.X_OK) else None
    found = shutil.which("node") or shutil.which("nodejs")
    if found:
        return found
    for raw in _NODE_FALLBACKS:
        candidate = Path(raw).expanduser()
        if os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _run_bridge(root: Path, args: list[str], timeout: int) -> dict[str, Any]:
    node = _node()
    if node is None:
        override = os.environ.get("TMF_NODE")
        detail = (
            f"TMF_NODE={override} is not an executable"
            if override
            else "node is not on PATH (a gateway run as a service does not "
            "inherit the login shell's PATH); set TMF_NODE to an absolute "
            "node path, or add node's directory to the gateway's PATH"
        )
        return {"ok": False, "code": "node-unavailable", "error": detail}
    env = dict(os.environ)
    env["TMF_FOUNDRY_ROOT"] = str(root)
    try:
        proc = subprocess.run(
            [node, str(_BRIDGE), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": "bridge-timeout", "error": f"bridge timed out after {timeout}s"}
    line = ""
    for candidate in reversed(proc.stdout.splitlines()):
        if candidate.strip():
            line = candidate.strip()
            break
    if not line:
        return {
            "ok": False,
            "code": "bridge-no-output",
            "error": (proc.stderr or "").strip()[:2000] or f"exit {proc.returncode} with no output",
        }
    try:
        return json.loads(line)
    except json.JSONDecodeError as err:
        return {"ok": False, "code": "bridge-bad-json", "error": f"{err}: {line[:500]}"}


def _make_handler(root: Path, tool_name: str):
    def handler(args: Any = None, **kwargs: Any) -> str:
        # Hosts differ: some dispatch handler(args_dict), others handler(**kwargs).
        # Accept both and merge, keeping explicit keywords authoritative.
        payload: dict[str, Any] = {}
        if isinstance(args, dict):
            payload.update(args)
        payload.update(kwargs)
        result = _run_bridge(root, ["call", tool_name, json.dumps(payload)], _TIMEOUT_CALL)
        envelope = result.get("envelope") if isinstance(result, dict) else None
        if isinstance(envelope, dict) and isinstance(envelope.get("content"), str):
            return envelope["content"]
        return json.dumps(result, ensure_ascii=False)

    handler.__name__ = f"tmf_handler_{tool_name}"
    return handler


def _log(ctx, message: str) -> None:
    """Report a mount decision without assuming the host exposes ``ctx.log``.

    Hosts differ on whether ``PluginContext`` carries a ``log`` helper, and a
    missing attribute here would raise out of ``register`` and make the host
    roll back every tool this plugin just registered.
    """
    hook = getattr(ctx, "log", None)
    if callable(hook):
        try:
            hook(message)
            return
        except Exception:  # pragma: no cover - host logger misbehaving
            pass
    logger.info("%s", message)


def register(ctx) -> None:
    root = _foundry_root()
    if root is None:
        _log(ctx, 
            "taskforce-model-foundry: no Foundry checkout found "
            "(set TMF_FOUNDRY_ROOT or place it at <plugin>/foundry); no tools registered"
        )
        return
    probe = _run_bridge(root, ["specs"], _TIMEOUT_SPECS)
    if not probe.get("ok"):
        _log(ctx, 
            "taskforce-model-foundry: adapter probe failed "
            f"({probe.get('code')}: {probe.get('error')}); no tools registered"
        )
        return
    registered = 0
    for spec in probe.get("specs") or []:
        fn = spec.get("function") if isinstance(spec, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        name = fn["name"]
        ctx.register_tool(
            name=name,
            toolset=TOOLSET,
            schema=fn.get("parameters") or {"type": "object", "properties": {}},
            handler=_make_handler(root, name),
            description=fn.get("description", ""),
            emoji="🏭",
        )
        registered += 1
    _log(ctx, 
        f"taskforce-model-foundry: registered {registered} tool(s) from {root} "
        f"(contract {probe.get('contractVersion')})"
    )
