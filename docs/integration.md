# Integration and installation (OpenClaw / Hermes)

This document is the host-side contract for mounting Taskforce Model Foundry
inside an agent runtime. Both adapters are **in-process ES modules** — no
daemon, no socket, no shelling out to the `tmf` CLI — and both are driven by the
single action table in `src/integrations/openclaw.mjs`, so a new action appears
on both hosts at once and the two hosts cannot drift.

Verified against the packaged artifact, not the source tree: see
`artifacts/taskforce-model-foundry/evidence/cp23-r7-pack-install.md` for the
`npm pack` → clean-directory `npm i` → in-process adapter run that produced the
numbers quoted here.

## 1. Install

The package has **zero runtime dependencies** and needs Node.js >= 20
(`tmf doctor` enforces the floor). Nothing in an install or a mount touches the
network.

```bash
# from the npm registry
npm i taskforce-model-foundry

# from a local tarball, which is how the install path was verified
npm pack                     # -> taskforce-model-foundry-0.2.0.tgz
cd /path/to/host
npm i /path/to/taskforce-model-foundry-0.2.0.tgz
```

The install provides one bin, `tmf` (`node_modules/.bin/tmf`), and the
export specifiers below. `package.json` is a fifth, deliberate `exports` entry
(see §3.1); it is not an API surface.

| Specifier | Purpose |
| --- | --- |
| `taskforce-model-foundry` | Root export; re-exports the API plus `.openclaw` / `.hermes`. |
| `taskforce-model-foundry/api` | Programmatic API (`doctor`, `status`, `synthesize`, `train`, `evaluate`, `exportModel`). |
| `taskforce-model-foundry/integrations/openclaw` | OpenClaw tool-provider adapter. |
| `taskforce-model-foundry/integrations/openclaw-plugin` | OpenClaw host plugin entry; needs the host-provided `openclaw` package. |
| `taskforce-model-foundry/integrations/openclaw-plugin/tools` | Host-free tool face (`buildTools`), importable without `openclaw`. |
| `taskforce-model-foundry/integrations/hermes` | Hermes tool-call adapter. |

The root export and the subpaths resolve to the **same module instances**
(`root.hermes.toolSpecs === hermes.toolSpecs`), so a host may mix specifiers
without ending up with two copies of the action table.

Smoke-check an install before mounting it:

```bash
./node_modules/.bin/tmf doctor      # exit 0 and "tmf doctor — OK"
```

`doctor` warns (it does not fail) when no config file is present and when the
optional `hf-peft` trainer backend is unavailable; the built-in JS trainer
needs nothing extra.

## 2. Mounting on OpenClaw

An OpenClaw host passes an action name plus a params object and gets a
JSON-serializable envelope back. `invoke` never throws.

```js
import { register, manifest } from 'taskforce-model-foundry/integrations/openclaw';

// `host.registerToolProvider` is called for you when the host exposes it.
const adapter = register(host, {
  baseDir: '/srv/foundry-workspace',   // where jobs/runs/exports live
  configFile: null,                    // default config for every action
  allowCompute: true,                  // false = read-only mount, see §4
});

manifest();                            // static, side-effect free
await adapter.invoke('doctor', {});
```

Envelopes:

```js
{ ok: true,  contractVersion: 1, action, result }
{ ok: false, contractVersion: 1, action, error: { code, message } }
```

## 3. Mounting on Hermes

A Hermes gateway hands over a flat prefixed tool name and a JSON **string** of
arguments, and shows the model whatever text comes back. `callTool` therefore
always returns a non-empty `content` string, success or failure.

```js
import { register, toolSpecs } from 'taskforce-model-foundry/integrations/hermes';

const adapter = register(host, { baseDir: '/srv/foundry-workspace', allowCompute: false });

toolSpecs();                                   // OpenAI-style function specs
await adapter.callTool('tmf_status', '{"jobId":"job-1"}');
```

`register` adapts to either gateway shape: `host.registerTool({ spec, handler })`
per tool when available, otherwise `host.registerTools({ tools, handler })`.

Envelopes:

```js
{ ok: true,  contractVersion: 1, host: 'hermes', tool, action, result, content }
{ ok: false, contractVersion: 1, host: 'hermes', tool, error: { code, message }, content: 'ERROR <code>: <message>' }
```

Every tool schema is closed (`additionalProperties: false`) and every parameter
carries a description, so a model cannot invent a param that the adapter would
then have to reject.

### 3.1 As a Hermes directory plugin

`integrations/hermes-plugin/taskforce_model_foundry/` mounts the same adapter
without any gateway code: Hermes discovers a directory plugin by `plugin.yaml`
plus a `register(ctx)`, and the plugin asks the Node bridge for the adapter's
own tool specs, so an action added on the Node side appears on Hermes with no
Python change.

The plugin directory ships in the npm tarball, so the copy works from either an
installed package or a source checkout; only the source root differs. Point
`TMF_PKG` at it once and the rest of the snippet is identical:

```bash
# npm install: ask node where the package landed (works under any node_modules depth)
TMF_PKG=$(node -p "require('path').dirname(require.resolve('taskforce-model-foundry/package.json'))")
# source checkout: TMF_PKG=/path/to/taskforce-model-foundry

cp -r "$TMF_PKG"/integrations/hermes-plugin/taskforce_model_foundry ~/.hermes/plugins/
ln -s "$TMF_PKG" ~/.hermes/plugins/taskforce_model_foundry/foundry
# or: export TMF_FOUNDRY_ROOT="$TMF_PKG"
```

`require.resolve` is why `"./package.json"` is an explicit `exports` entry: a
closed `exports` map hides the manifest, and the host then has no supported way
to locate the installed plugin source.

Checkout resolution order: `TMF_FOUNDRY_ROOT`, then `<plugin dir>/foundry`,
then `~/.hermes/taskforce-model-foundry`. `node` resolution order:
`TMF_NODE`, then the gateway's PATH, then `~/.local/bin/node`,
`/usr/local/bin/node`, `/opt/homebrew/bin/node`, `/usr/bin/node`.

Three host details are easy to get wrong, and all three were observed on a
live gateway:

- **The gateway's PATH is not your shell's PATH.** A gateway started as a
  service (systemd, launchd, a supervisor) gets a cropped environment, so a
  per-user node — nvm, or `~/.local/bin/node` — is invisible to the plugin
  even though `node -v` works in your terminal. The probe then fails with
  `node-unavailable` and the mount logs `no tools registered`: the plugin is
  loaded and the toolset is simply empty, which looks like `Unknown tool:
  tmf_export` at call time rather than like an install error. The fallback
  locations above cover the common installs; for anything else set
  `TMF_NODE=/absolute/path/to/node` in the gateway's environment (not in your
  shell profile) and reload the gateway. To tell this apart from the
  hyphen/underscore mistake below, check the mount line: a PATH problem logs
  `adapter probe failed (node-unavailable: ...)`, a not-enabled plugin logs
  nothing at all.

- **Enable the plugin by its manifest `name`, not its Python package name.**
  The directory is `taskforce_model_foundry` (underscores, importable) but
  `plugins.enabled` is matched against `plugin.yaml`'s `name`, so the entry is
  `taskforce-model-foundry` with hyphens. The underscore spelling is silently
  not enabled.
- **A successful registration does not prove the tools are callable.** Hermes'
  `tools.registry.dispatch` invokes `entry.handler(args, **kwargs)` — the
  argument dict goes in *positionally* — while other hosts call
  `handler(**kwargs)`. The plugin's handlers accept both shapes and merge them
  (explicit keywords win); a keyword-only handler registers fine and then
  raises `TypeError` on the first real tool call. Likewise `ctx.log` is not
  guaranteed to exist, and anything raised out of `register` rolls back every
  tool the plugin just registered, so the mount logs defensively and reports a
  missing checkout or a failed bridge probe as a skipped mount instead.

Both behaviours are pinned by `integrations/hermes-plugin/tests/test_register.py`,
which `npm test` runs through `test/integration-hermes-plugin.test.mjs` (skipped
when no Python interpreter is on PATH).

## 4. Action and tool table

Eleven actions, eleven tools, one table. `compute: true` means the call spends real
CPU time and writes run artifacts; those are the calls a host should gate.

| OpenClaw action | Hermes tool | Compute | Required | Accepted params |
| --- | --- | --- | --- | --- |
| `doctor` | `tmf_doctor` | no | — | `configFile`, `requireConfig` |
| `status` | `tmf_status` | no | `jobId` | `jobId`, `configFile`, `auditTail` |
| `synthesize` | `tmf_synthesize` | yes | — | `outDir`, `seed`, `count`, `profile` |
| `train` | `tmf_train` | yes | — | `configFile`, `jobId`, `resume`, `resumeCancelled`, `auditEveryStep`, `background`, `retry` |
| `train_status` | `tmf_train_status` | no | `jobId`, `requestId` | `configFile`, `jobId`, `requestId` |
| `train_cancel` | `tmf_train_cancel` | yes | `jobId`, `requestId` | `configFile`, `jobId`, `requestId` |
| `evaluate` | `tmf_evaluate` | yes | — | `configFile`, `jobId`, `gates`, `reportFile`, `maxNewTokens`, `background`, `retry` |
| `evaluate_status` | `tmf_evaluate_status` | no | `jobId`, `requestId` | `configFile`, `jobId`, `requestId` |
| `screenplay_quality_prepare` | `tmf_screenplay_quality_prepare` | yes | `jobId`, `evalReport`, `suiteFile`, `baseOutputs`, `daptOutputs`, `candidateOutputs`, `cleanDaptReport`, `cleanDaptModel`, `outDir` | `configFile`, `jobId`, `evalReport`, `suiteFile`, `scriptsDir`, `baseOutputs`, `daptOutputs`, `candidateOutputs`, `cleanDaptReport`, `cleanDaptModel`, `outDir` |
| `screenplay_quality_finalize` | `tmf_screenplay_quality_finalize` | yes | `stageFile`, `judgeVerdicts`, `cleanCandidateReport`, `cleanCandidateModel`, `reportFile` | `configFile`, `stageFile`, `judgeVerdicts`, `cleanCandidateReport`, `cleanCandidateModel`, `reportFile` |
| `export` | `tmf_export` | yes | — | `configFile`, `jobId`, `evalReport`, `qualityReport`, `outDir`, `version`, `evaluateIfMissing` |

**Read-only mount.** `allowCompute: false` keeps the adapter useful for
inspection while making compute impossible: the four read-only actions answer
normally, and all seven compute actions are refused with
`compute-not-authorized` *before* any handler runs — nothing is written, no
partial job appears. Mount read-only wherever the host cannot itself vouch that
training was authorized.

## 5. Error codes

Stable codes, shared by both adapters, returned as data rather than thrown.
A host can branch on `error.code` without parsing messages.

| Code | Meaning |
| --- | --- |
| `invalid-action` | OpenClaw only: action was not a non-empty string. |
| `unknown-action` | OpenClaw only: action name is not in the table. |
| `unknown-tool` | Hermes only: tool name is unprefixed or unknown. |
| `invalid-json` | Hermes only: the argument string was not valid JSON. |
| `invalid-params` | Arguments did not decode to a plain object. |
| `unknown-params` | A param outside the action's accepted list was supplied. |
| `missing-params` | A required param (today only `jobId`) was absent or null. |
| `compute-not-authorized` | Compute action on an `allowCompute: false` mount. |
| `config-error` | Config file missing, unparsable or internally inconsistent. |
| `action-failed` | Fallback for any other handler failure; message carries the detail. |

Handler-specific codes pass through unchanged when the underlying API raises an
error carrying its own `code`, so this table is a floor, not a closed set — but
`ok: false` plus a string `error.code` is guaranteed in every case.

## 6. Contract versioning

`CONTRACT_VERSION` (currently `1`) appears in every envelope and both
manifests. It is bumped only on a breaking change to the action set or the
envelope shape; additive actions and additive params do not bump it. A host
should treat an unexpected major as "do not mount" rather than degrade.
