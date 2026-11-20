// Trainer backend registry.
//
// A backend is the seam between the foundry and whatever actually updates
// weights. Everything above this layer (jobs, evaluation, export, integrations)
// talks only to this interface, which is what makes a second backend a drop-in
// rather than a rewrite.
//
// Required surface:
//
//   id             stable identifier used in configs and lineage
//   capabilities   { modes, frameworks, devices, supportsResume }
//   probe()        -> { available, reason?, details? }   must never throw
//   createModel()  -> handle for a fresh model
//   loadModel()    -> handle restored from a checkpoint
//   saveModel()    -> { file, sha256 }
//   train()        -> { steps, history, stopReason, metrics }
//   evaluate()     -> { loss, perplexity, tokens }
//   generate()     -> string
//
// `probe` is deliberately total: an unavailable backend is a recorded fact, not
// an exception, so a job can report "not verified in this environment" instead
// of failing in a way that looks like a bug.

const REQUIRED_METHODS = ['probe', 'createModel', 'loadModel', 'saveModel', 'train', 'evaluate', 'generate'];

const registry = new Map();

export function registerBackend(backend) {
  if (!backend?.id) throw new Error('a backend must expose an id');
  for (const method of REQUIRED_METHODS) {
    if (typeof backend[method] !== 'function') {
      throw new Error(`backend "${backend.id}" is missing required method ${method}()`);
    }
  }
  if (!backend.capabilities?.modes?.length) {
    throw new Error(`backend "${backend.id}" must declare capabilities.modes`);
  }
  registry.set(backend.id, backend);
  return backend;
}

export function getBackend(id) {
  const backend = registry.get(id);
  if (!backend) {
    throw new Error(`unknown trainer backend "${id}"; registered: ${[...registry.keys()].join(', ') || 'none'}`);
  }
  return backend;
}

export function listBackends() {
  return [...registry.values()].map((b) => ({ id: b.id, capabilities: b.capabilities }));
}

export async function probeBackends() {
  const out = [];
  for (const backend of registry.values()) {
    let result;
    try {
      result = await backend.probe();
    } catch (err) {
      // A throwing probe is itself a availability failure, not a crash.
      result = { available: false, reason: `probe threw: ${err.message}` };
    }
    out.push({ id: backend.id, capabilities: backend.capabilities, ...result });
  }
  return out;
}

export async function resolveBackend(id) {
  const backend = getBackend(id);
  const probe = await backend.probe();
  if (!probe.available) {
    const err = new Error(`trainer backend "${id}" is not available in this environment: ${probe.reason}`);
    err.backendProbe = probe;
    err.code = 'BACKEND_UNAVAILABLE';
    throw err;
  }
  return backend;
}

export function clearBackends() {
  registry.clear();
}

// Side-effecting imports register the built-in backends.
export async function loadDefaultBackends() {
  const builtin = await import('./builtin-js.mjs');
  const hf = await import('./hf-peft.mjs');
  registerBackend(builtin.default);
  registerBackend(hf.default);
  return listBackends();
}
