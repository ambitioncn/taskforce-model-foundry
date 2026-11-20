"""Host-compatibility regression tests for the Hermes directory plugin.

Two real breakages found while mounting this plugin on a live Hermes host are
pinned here, because both surfaced only at dispatch/registration time and
neither is visible from the Node side of the adapter:

  1. ``tools.registry.dispatch`` calls ``entry.handler(args, **kwargs)`` with
     the argument dict *positionally*, while other hosts call
     ``handler(**kwargs)``. A keyword-only handler raised ``TypeError`` on the
     very first dispatch even though registration had succeeded.
  2. ``PluginContext`` does not always carry a ``log`` helper, and an
     ``AttributeError`` raised out of ``register`` makes the host roll back
     every tool the plugin just registered.

The Node bridge is stubbed: these tests are about the Python mount contract,
not about Foundry actions (those are covered by test/integration-hermes.test.mjs).
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import taskforce_model_foundry as plugin  # noqa: E402


SPECS = {
    "ok": True,
    "contractVersion": "test",
    "specs": [
        {
            "function": {
                "name": "tmf_status",
                "description": "status",
                "parameters": {"type": "object", "properties": {"jobId": {"type": "string"}}},
            }
        },
        {"function": {"name": "tmf_doctor", "description": "doctor", "parameters": {}}},
        {"not_a_function": True},
    ],
}


class RecordingBridge:
    """Stands in for ``_run_bridge`` and records every invocation."""

    def __init__(self, specs=None):
        self.calls = []
        self._specs = specs if specs is not None else SPECS

    def __call__(self, root, args, timeout):
        self.calls.append({"root": root, "args": list(args), "timeout": timeout})
        if args[:1] == ["specs"]:
            return self._specs
        return {"ok": True, "envelope": {"content": "bridge-content"}}


class CtxWithoutLog:
    """A host context that exposes only ``register_tool`` (no ``log``)."""

    def __init__(self):
        self.tools = {}

    def register_tool(self, *, name, toolset, schema, handler, description="", emoji=""):
        self.tools[name] = {
            "toolset": toolset,
            "schema": schema,
            "handler": handler,
            "description": description,
            "emoji": emoji,
        }


class CtxWithLog(CtxWithoutLog):
    def __init__(self):
        super().__init__()
        self.messages = []

    def log(self, message):
        self.messages.append(message)


class PluginMountTest(unittest.TestCase):
    def setUp(self):
        self._real_bridge = plugin._run_bridge
        self._real_root = plugin._foundry_root
        self.bridge = RecordingBridge()
        plugin._run_bridge = self.bridge
        plugin._foundry_root = lambda: Path("/fake/foundry")

    def tearDown(self):
        plugin._run_bridge = self._real_bridge
        plugin._foundry_root = self._real_root

    def _register(self, ctx):
        plugin.register(ctx)
        return ctx

    def test_register_without_ctx_log(self):
        ctx = self._register(CtxWithoutLog())
        self.assertEqual(sorted(ctx.tools), ["tmf_doctor", "tmf_status"])
        self.assertEqual(ctx.tools["tmf_status"]["toolset"], plugin.TOOLSET)

    def test_dispatch_conventions_agree(self):
        ctx = self._register(CtxWithoutLog())
        handler = ctx.tools["tmf_status"]["handler"]

        self.bridge.calls.clear()
        positional = handler({"jobId": "job-1"})
        keyword = handler(jobId="job-1")

        self.assertEqual(positional, "bridge-content")
        self.assertEqual(keyword, "bridge-content")
        self.assertEqual(len(self.bridge.calls), 2)
        payloads = [json.loads(call["args"][2]) for call in self.bridge.calls]
        self.assertEqual(payloads[0], {"jobId": "job-1"})
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(self.bridge.calls[0]["args"][:2], ["call", "tmf_status"])

    def test_no_arg_dispatch_sends_empty_payload(self):
        ctx = self._register(CtxWithoutLog())
        self.bridge.calls.clear()
        ctx.tools["tmf_doctor"]["handler"]()
        self.assertEqual(json.loads(self.bridge.calls[0]["args"][2]), {})

    def test_explicit_keywords_win_over_positional_dict(self):
        ctx = self._register(CtxWithoutLog())
        handler = ctx.tools["tmf_status"]["handler"]
        self.bridge.calls.clear()
        handler({"jobId": "from-dict", "keep": 1}, jobId="from-kwargs")
        self.assertEqual(
            json.loads(self.bridge.calls[0]["args"][2]),
            {"jobId": "from-kwargs", "keep": 1},
        )

    def test_non_envelope_result_is_returned_as_json(self):
        ctx = self._register(CtxWithoutLog())
        plugin._run_bridge = lambda root, args, timeout: {"ok": False, "code": "boom"}
        out = ctx.tools["tmf_doctor"]["handler"]()
        self.assertEqual(json.loads(out), {"ok": False, "code": "boom"})

    def test_missing_checkout_registers_nothing_and_does_not_raise(self):
        plugin._foundry_root = lambda: None
        ctx = self._register(CtxWithLog())
        self.assertEqual(ctx.tools, {})
        self.assertTrue(any("no tools registered" in m for m in ctx.messages))

    def test_failed_probe_registers_nothing_and_does_not_raise(self):
        plugin._run_bridge = RecordingBridge(specs={"ok": False, "code": "bridge-timeout", "error": "x"})
        ctx = self._register(CtxWithLog())
        self.assertEqual(ctx.tools, {})
        self.assertTrue(any("adapter probe failed" in m for m in ctx.messages))

    def test_raising_ctx_log_does_not_break_registration(self):
        class CtxBadLog(CtxWithoutLog):
            def log(self, message):
                raise RuntimeError("host logger down")

        ctx = self._register(CtxBadLog())
        self.assertEqual(sorted(ctx.tools), ["tmf_doctor", "tmf_status"])


class NodeResolutionTest(unittest.TestCase):
    """A gateway run as a service has a cropped PATH, so node must still resolve.

    This is the third live-host breakage: the plugin loaded, the registry scope
    was right, and the toolset was empty, because ``shutil.which("node")``
    found nothing under a service PATH while ``node -v`` worked in the
    operator's shell. The symptom reaches the model as ``Unknown tool:
    tmf_export``, which does not point at PATH at all, so the resolution order
    and the message that explains it are pinned here.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake_node = Path(self.tmp.name) / "node"
        self.fake_node.write_text("#!/bin/sh\nexit 0\n")
        self.fake_node.chmod(self.fake_node.stat().st_mode | stat.S_IXUSR)

    def test_tmf_node_overrides_path(self):
        with mock.patch.dict(os.environ, {"TMF_NODE": str(self.fake_node)}):
            self.assertEqual(plugin._node(), str(self.fake_node))

    def test_tmf_node_expands_user(self):
        # Operators write ~/.local/bin/node; an unexpanded "~" would just miss.
        with mock.patch.dict(os.environ, {"TMF_NODE": "~/node", "HOME": self.tmp.name}):
            self.assertEqual(plugin._node(), str(self.fake_node))

    def test_non_executable_tmf_node_is_rejected(self):
        dud = Path(self.tmp.name) / "not-node"
        dud.write_text("")
        with mock.patch.dict(os.environ, {"TMF_NODE": str(dud)}):
            self.assertIsNone(plugin._node())

    def test_fallback_locations_are_tried_when_path_is_cropped(self):
        env = dict(os.environ)
        env.pop("TMF_NODE", None)
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(plugin.shutil, "which", return_value=None), \
             mock.patch.object(plugin, "_NODE_FALLBACKS", ("/nope/node", str(self.fake_node))):
            self.assertEqual(plugin._node(), str(self.fake_node))

    def test_node_unavailable_error_names_tmf_node(self):
        env = dict(os.environ)
        env.pop("TMF_NODE", None)
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(plugin, "_node", return_value=None):
            result = plugin._run_bridge(Path("/fake/foundry"), ["specs"], 1)
        self.assertEqual(result["code"], "node-unavailable")
        self.assertIn("TMF_NODE", result["error"])
        self.assertIn("PATH", result["error"])

    def test_bad_tmf_node_error_names_the_bad_value(self):
        with mock.patch.dict(os.environ, {"TMF_NODE": "/nope/node"}):
            result = plugin._run_bridge(Path("/fake/foundry"), ["specs"], 1)
        self.assertEqual(result["code"], "node-unavailable")
        self.assertIn("/nope/node", result["error"])


if __name__ == "__main__":
    unittest.main()
