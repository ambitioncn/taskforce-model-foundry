# taskforce-model-foundry

An automated model customization engine for OpenClaw and Hermes: it generates or
ingests a corpus, runs **real** post-training (weights actually move), evaluates
the result on an independent holdout split, and refuses to call a regression an
improvement.

- Node.js 20+, **zero runtime dependencies**
- Real training, not a simulation: every job reports its parameter L2 delta
- Durable, resumable, audited jobs — state is re-read from disk, not memory
- Gate-guarded evaluation: `assertDeliverable()` throws on a failed gate
- No private data and no network needed to try it out

## Quickstart

```bash
git clone https://github.com/ambitioncn/taskforce-model-foundry.git
cd taskforce-model-foundry
node bin/tmf.mjs --version
node bin/tmf.mjs init -C /tmp/my-foundry
node bin/tmf.mjs train -C /tmp/my-foundry --job first
node bin/tmf.mjs status first -C /tmp/my-foundry
```

This release is distributed as source on GitHub; an npm registry release has not
been published. The built-in CPU backend needs no npm dependencies.

See the full walkthrough in **[docs/tutorial.md](docs/tutorial.md)**, the host-side
contract in **[docs/integration.md](docs/integration.md)**, or watch the
whole chain run end to end:

```bash
# from an install (the script ships in the tarball)
bash node_modules/taskforce-model-foundry/scripts/e2e-demo.sh

# from a source checkout
npm run demo              # bash scripts/e2e-demo.sh
```

The demo writes to a temporary workdir and removes it again, so it is safe to
run against an installed copy.

## Programmatic API

```js
import { doctor, synthesize, train, status } from 'taskforce-model-foundry';
```

The CLI is a thin wrapper over this API, so embedders get identical behaviour
without shelling out. Lower-level building blocks are exported under the stable
namespaces `data`, `synth`, `jobs`, `trainJob`, `evalJob`, `gates`, `backends`.

## CLI surface

| Command | Purpose |
| --- | --- |
| `tmf doctor` | Check environment, backends and config (`--json` for CI) |
| `tmf init` | Write a starter `tmf.config.json` plus synthetic data |
| `tmf config validate` | Report *every* config problem with its field path |
| `tmf backends` | List trainer backends and their availability |
| `tmf synth` | Generate a synthetic corpus |
| `tmf train` | Run a real post-training job (resumable) |
| `tmf status <jobId>` | Durable job state and recent audit events |

Exit codes: `0` ok, `1` check/job failed, `2` usage error.

`tmf doctor` grades an absent or unreadable config as a *warning* and still
exits `0`, so the environment and backend verdict survives a project that has
no config yet. Point it at the file with `--config <file>`, or use `tmf config
validate`, when CI must fail on a bad config.

## Scope and authority

Synthetic data and local CPU training are the default. Real private company
data, paid training runs, production changes and publishing are each separately
authorized operations and are never performed implicitly.

## Development

```bash
npm test                  # the full suite
```

MIT licensed.
