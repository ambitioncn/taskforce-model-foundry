// Second trainer backend: Hugging Face transformers + PEFT via a local Python
// process.
//
// This backend exists to keep the registry interface honest. It is implemented
// against the same contract as the builtin trainer, discovers its dependencies
// at runtime, and reports precisely what is missing when they are absent. It
// never falls back to the builtin trainer and never fabricates metrics: if the
// environment cannot run it, `probe()` says so and callers record the
// capability as unverified rather than claiming a result.
//
// Execution goes through `hf_peft_runner.py`, a checked-in program, so the code
// that actually trains is reviewable rather than assembled from strings at call
// time. `renderTrainingScript()` is retained and now emits the exact spec and
// command line that will be executed.
//
// This is a local-execution backend only. It does not download models and does
// not call a hosted API, so it stays inside the project's "no paid training,
// no external data egress" constraints. A caller must point `modelId` at a
// model already present on local disk; the runner sets HF offline mode so a
// missing local model fails instead of silently fetching one.

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { writeFileAtomic, ensureDir, readJson } from '../../core/fsx.mjs';
import { sha256File } from '../../core/hash.mjs';

// `datasets` and `accelerate` are imported by the training path, so a probe
// that omitted them would report "available" for an environment that cannot
// train. They are part of the required set.
export const REQUIRED_PYTHON_PACKAGES = ['torch', 'transformers', 'peft', 'datasets', 'accelerate'];

const RUNNER = path.join(path.dirname(fileURLToPath(import.meta.url)), 'hf_peft_runner.py');
const RESULT_PREFIX = '__TMF_RESULT__';

// Defaults kept small on purpose: this backend is expected to run on a CPU box
// next to the rest of the toolchain, not on a training cluster.
// This is a virtual-address-space cap, not resident RAM. Importing CUDA-enabled
// PyTorch can reserve more than 4 GiB of address space before model loading;
// keep the explicit caller override for tighter bounded jobs.
const DEFAULT_LIMITS = { maxMemoryMb: 65536, threads: 1, maxSeqLen: 128 };

function pythonBin() {
  return process.env.TMF_PYTHON || 'python3';
}

function runPython(args, { timeoutMs = 600000, cwd, signal } = {}) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(pythonBin(), args, { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
    } catch (err) {
      resolve({ code: -1, stdout: '', stderr: err.message, spawnFailed: true });
      return;
    }
    let stdout = '';
    let stderr = '';
    let cancelled = false;
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
      stderr += `\ntimed out after ${timeoutMs}ms`;
    }, timeoutMs);
    // Cancellation is a SIGKILL of the worker process, not a request the
    // worker may ignore: the kernel tears it down. In-flight work is lost,
    // which is why the only recoverable state is the last written checkpoint.
    const onAbort = () => {
      cancelled = true;
      child.kill('SIGKILL');
    };
    if (signal) {
      if (signal.aborted) onAbort();
      else signal.addEventListener('abort', onAbort, { once: true });
    }
    child.stdout.on('data', (c) => { stdout += c; });
    child.stderr.on('data', (c) => { stderr += c; });
    child.on('error', (err) => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      resolve({ code: -1, stdout, stderr: `${stderr}${err.message}`, spawnFailed: true });
    });
    child.on('close', (code, closeSignal) => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      resolve({ code, signal: closeSignal, stdout, stderr, cancelled, timedOut });
    });
  });
}

const PROBE_SCRIPT = `
import json, importlib
report = {}
for name in ${JSON.stringify(REQUIRED_PYTHON_PACKAGES)}:
    try:
        mod = importlib.import_module(name)
        report[name] = getattr(mod, "__version__", "unknown")
    except Exception as exc:
        report[name] = None
try:
    import torch
    report["_devices"] = ["cuda" if torch.cuda.is_available() else "cpu"]
except Exception:
    report["_devices"] = []
print(json.dumps(report))
`;

function unavailable(method, probe) {
  const err = new Error(
    `hf-peft backend cannot ${method}: ${probe?.reason ?? 'environment not available'}. `
    + 'Install the missing packages locally, or run with backend "builtin-js".',
  );
  err.code = 'BACKEND_UNAVAILABLE';
  err.backendProbe = probe;
  return err;
}

function parseResult(result, op) {
  const line = result.stdout.split('\n').reverse().find((l) => l.startsWith(RESULT_PREFIX));
  if (!line) {
    throw new Error(
      `hf-peft ${op} produced no result line; last stderr: ${(result.stderr || '').trim().slice(-400)}`,
    );
  }
  return JSON.parse(line.slice(RESULT_PREFIX.length).trim());
}

async function writeSpec(spec, { outDir, op }) {
  await ensureDir(outDir);
  const file = path.join(outDir, `hf-peft-${op}-spec.json`);
  await writeFileAtomic(file, `${JSON.stringify(spec, null, 2)}\n`);
  return file;
}

function limitsFrom(budget = {}, overrides = {}) {
  const envMemory = process.env.TMF_HF_MAX_MEMORY_MB;
  if (envMemory !== undefined && (!Number.isInteger(Number(envMemory)) || Number(envMemory) <= 0)) {
    throw new Error('TMF_HF_MAX_MEMORY_MB must be a positive integer');
  }
  return {
    maxMemoryMb: overrides.maxMemoryMb ?? budget.maxMemoryMb ?? (envMemory ? Number(envMemory) : DEFAULT_LIMITS.maxMemoryMb),
    maxCpuSeconds: overrides.maxCpuSeconds ?? budget.maxCpuSeconds ?? null,
    threads: overrides.threads ?? budget.threads ?? DEFAULT_LIMITS.threads,
    maxSeqLen: overrides.maxSeqLen ?? DEFAULT_LIMITS.maxSeqLen,
  };
}

async function runOp(op, spec, { outDir, timeoutMs, signal }) {
  const specFile = await writeSpec({ op, ...spec }, { outDir, op });
  const result = await runPython([RUNNER, specFile], { timeoutMs, signal });
  if (result.cancelled) {
    const err = new Error(`hf-peft ${op} was cancelled (worker killed with SIGKILL)`);
    err.code = 'BACKEND_CANCELLED';
    err.specFile = specFile;
    throw err;
  }
  if (result.code !== 0) {
    // RLIMIT_AS surfaces as a MemoryError / allocation failure inside the
    // worker; RLIMIT_CPU kills it with SIGKILL. Both are reported as the
    // resource failure they are instead of a generic crash.
    const stderr = (result.stderr || '').trim();
    // A too-small RLIMIT_AS can bite before training starts: the framework's
    // own shared objects fail to mmap. That is still the memory bound doing
    // its job, so it is classified as one rather than as a generic crash.
    const memoryBound = /MemoryError|Cannot allocate memory|bad_alloc|DefaultCPUAllocator|failed to map segment|Cannot mmap|OutOfMemoryError/i.test(stderr);
    const err = new Error(
      `hf-peft ${op} failed (exit ${result.code}${result.signal ? `, signal ${result.signal}` : ''})`
      + `${memoryBound ? ' due to the configured memory limit' : ''}: ${stderr.slice(-500)}`,
    );
    err.code = /TMF_TRAIN_TOKEN_BUDGET_EXCEEDED/.test(stderr) ? 'BACKEND_TOKEN_BUDGET'
      : memoryBound ? 'BACKEND_MEMORY_LIMIT'
      : result.signal === 'SIGKILL' ? 'BACKEND_RESOURCE_LIMIT'
        : result.timedOut ? 'BACKEND_TIMEOUT' : 'BACKEND_FAILED';
    err.specFile = specFile;
    err.stderrTail = stderr.slice(-500);
    throw err;
  }
  return { payload: parseResult(result, op), specFile, stderr: result.stderr };
}

// Weight/config files a HF model or PEFT adapter directory is made of. Other
// files (trainer state, logs, tokenizer caches) are deliberately excluded: they
// move between runs that produce identical weights.
const FINGERPRINT_EXTENSIONS = ['.safetensors', '.bin', '.pt', '.json', '.model'];
const FINGERPRINT_MAX_DEPTH = 3;

function digestWeightDir(dir, depth = 0) {
  const out = [];
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    // A missing directory is a real state (a base-only handle), not an error:
    // it contributes nothing, and the modelId still distinguishes the handle.
    return out;
  }
  for (const entry of entries.sort((a, b) => (a.name < b.name ? -1 : 1))) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (depth >= FINGERPRINT_MAX_DEPTH) continue;
      for (const nested of digestWeightDir(full, depth + 1)) out.push(`${entry.name}/${nested}`);
      continue;
    }
    if (!entry.isFile()) continue;
    if (!FINGERPRINT_EXTENSIONS.includes(path.extname(entry.name))) continue;
    const buf = fs.readFileSync(full);
    out.push(`${entry.name}:${buf.length}:${crypto.createHash('sha256').update(buf).digest('hex')}`);
  }
  return out;
}

const backend = {
  id: 'hf-peft',

  capabilities: {
    modes: ['lora', 'full-finetune'],
    // The tokenizer belongs to the model, not to the job: ids are produced by
    // the HF tokenizer inside the Python worker. A job-built JS vocabulary
    // would produce ids the model's embedding table cannot interpret, so the
    // job must not encode for this backend. See prepareDataset().
    ownsTokenizer: true,
    frameworks: ['torch', 'transformers', 'peft', 'datasets'],
    devices: ['cpu', 'cuda'],
    // Resume restarts from the last Trainer checkpoint directory written under
    // the run's output dir, which is step-granular only at `checkpointEvery`
    // boundaries. Steps after the last checkpoint are replayed, not recovered.
    supportsResume: true,
    resumeGranularity: 'checkpoint-step',
    // Cancel kills the worker process. It is forced at the process level, but
    // it is not a graceful mid-step save: only already-written checkpoints
    // survive.
    supportsCancel: true,
    cancelSemantics: 'sigkill-worker',
    // Hard, kernel-enforced per-process bounds. This is a bound on the worker
    // process, not a security sandbox: it does not isolate the filesystem or
    // network.
    resourceBounds: ['max-steps', 'max-seconds', 'max-memory-mb', 'max-cpu-seconds', 'threads'],
    isolation: 'process-rlimit',
    precision: 'float32,bfloat16,nf4',
    requiresLocalModelCache: true,
  },

  async probe() {
    const dir = path.join(process.cwd(), '.tmf-run', 'probe');
    await ensureDir(dir);
    const script = path.join(dir, 'hf-peft-probe.py');
    await writeFileAtomic(script, PROBE_SCRIPT);
    const result = await runPython([script], { timeoutMs: 120000 });
    if (result.spawnFailed || result.code !== 0) {
      return {
        available: false,
        reason: `python interpreter unusable (${(result.stderr || '').trim().split('\n').slice(-1)[0] || `exit ${result.code}`})`,
        details: { python: pythonBin(), probeScript: script },
      };
    }
    let report;
    try {
      report = JSON.parse(result.stdout.trim().split('\n').slice(-1)[0]);
    } catch {
      return { available: false, reason: 'probe output was not valid json', details: { stdout: result.stdout.slice(0, 200) } };
    }
    const missing = REQUIRED_PYTHON_PACKAGES.filter((name) => !report[name]);
    if (missing.length > 0) {
      return {
        available: false,
        reason: `missing python packages: ${missing.join(', ')}`,
        details: { versions: report, missing, installHint: `pip install ${missing.join(' ')}` },
      };
    }
    return {
      available: true,
      reason: 'torch, transformers, peft, datasets and accelerate are importable',
      details: { versions: report, devices: report._devices, runner: RUNNER },
    };
  },

  // modelId may arrive either directly (direct backend callers) or inside
  // modelConfig (the training job passes one modelConfig through to whichever
  // backend is selected), so both spellings are accepted under one contract.
  async createModel({ modelId, outDir, modelConfig = {} } = {}) {
    const probe = await backend.probe();
    if (!probe.available) throw unavailable('create a model', probe);
    const resolvedId = modelId ?? modelConfig.modelId ?? modelConfig.baseModel ?? null;
    if (!resolvedId) {
      throw new Error('hf-peft requires modelId (or modelConfig.modelId) pointing at a locally cached model');
    }
    return {
      backend: backend.id,
      kind: 'base',
      modelId: resolvedId,
      mode: modelConfig.mode ?? 'lora',
      outDir: outDir ?? modelConfig.outDir ?? null,
      modelConfig,
      python: true,
    };
  },

  // Counterpart to builtin-js prepareDataset(): this backend keeps the records
  // as text and lets the Python worker tokenize them with the model's own
  // tokenizer. Records are validated and filtered here so an empty or
  // malformed split fails in Node with a readable error instead of inside the
  // worker. `tokens` is null on purpose: a real count needs the HF tokenizer,
  // and the worker reports the actual token count in its training metrics.
  prepareDataset({ records, maxSeqLen = 96 } = {}) {
    const examples = [];
    const skipped = [];
    for (const record of records ?? []) {
      const prompt = typeof record?.prompt === 'string' ? record.prompt : '';
      const completion = typeof record?.completion === 'string' ? record.completion : '';
      const text = typeof record?.text === 'string' ? record.text : '';
      // Whitespace is deliberately preserved. The worker derives the label mask
      // from len(tokenize(prompt)) against tokenize(prompt + completion), so
      // stripping the leading space off " eight" makes the boundary token
      // ("to" + "eight") re-merge, the supervised span collapse to empty, and
      // every step report loss 0 with zero parameter movement.
      if (prompt.trim() && completion.trim()) {
        examples.push({ prompt, completion });
      } else if (text.trim()) {
        examples.push({ text });
      } else {
        skipped.push({ reason: 'record has neither prompt+completion nor text' });
      }
    }
    return { examples, skipped, tokens: null, tokenizer: 'model-owned', maxSeqLen, vocabSize: null };
  },

  // attachAdapter mirrors the builtin backend: the evaluation job loads the
  // base checkpoint and attaches the adapter checkpoint so baseline and tuned
  // are two separate handles over the same base. Without honouring it here the
  // "tuned" handle would silently be the untouched base model.
  async loadModel({ file, attachAdapter = null } = {}) {
    const probe = await backend.probe();
    if (!probe.available) throw unavailable('load a checkpoint', probe);
    const payload = await readJson(file);
    if (payload.backend !== backend.id) throw new Error(`checkpoint backend "${payload.backend}" is not ${backend.id}`);
    if (attachAdapter) {
      const adapter = await readJson(attachAdapter);
      if (adapter.backend !== backend.id) {
        throw new Error(`adapter backend "${adapter.backend}" is not ${backend.id}`);
      }
      if (adapter.modelId !== payload.modelId) {
        throw new Error(`adapter base "${adapter.modelId}" does not match checkpoint base "${payload.modelId}"`);
      }
      const adapterDir = adapter.adapterDir ?? adapter.dir ?? null;
      if (!adapterDir) throw new Error('adapter checkpoint has no adapter directory to attach');
      return {
        backend: backend.id,
        kind: 'adapter',
        modelId: payload.modelId,
        dir: adapterDir,
        adapterDir,
        mode: adapter.mode ?? 'lora',
        modelConfig: adapter.modelConfig ?? payload.modelConfig ?? {},
        meta: { ...(adapter.meta ?? {}), attachedTo: file },
        file: attachAdapter,
      };
    }
    return {
      backend: backend.id,
      kind: payload.kind,
      modelId: payload.modelId,
      dir: payload.dir,
      adapterDir: payload.adapterDir ?? payload.dir ?? null,
      mode: payload.mode ?? 'lora',
      modelConfig: payload.modelConfig ?? {},
      meta: payload.meta ?? {},
      file,
    };
  },

  // Same contract as the builtin backend: a synchronous content digest of the
  // weights this handle points at, so callers can detect "adapter does not
  // belong to this base" without loading Python. Python weights live in
  // directories rather than one JSON file, so the digest is over the sorted
  // (relative path, size, sha256) triples of the weight/config files found
  // there. Directory walking is depth-bounded and extension-filtered so a
  // fingerprint cannot be turned into an unbounded scan of the filesystem.
  fingerprint(handle) {
    if (!handle?.modelId) throw new Error('hf-peft fingerprint() requires a handle with modelId');
    const hash = crypto.createHash('sha256');
    hash.update(`${backend.id}\u0000${handle.modelId}\u0000${handle.mode ?? 'lora'}`);
    const dirs = [];
    for (const dir of [handle.dir, handle.adapterDir]) {
      if (dir && !dirs.includes(dir)) dirs.push(dir);
    }
    // Positional label, not the directory name: identical weights must
    // fingerprint identically after a checkpoint directory is moved or copied.
    for (let i = 0; i < dirs.length; i += 1) {
      hash.update(`\u0000dir${i}`);
      for (const entry of digestWeightDir(dirs[i])) hash.update(`\u0000${entry}`);
    }
    return hash.digest('hex');
  },

  async saveModel({ handle, file, kind = 'adapter', meta = {} } = {}) {
    const probe = await backend.probe();
    if (!probe.available) throw unavailable('save a checkpoint', probe);
    const payload = {
      format: 'tmf-checkpoint-v1',
      backend: backend.id,
      kind,
      modelId: handle.modelId,
      dir: handle.dir ?? handle.adapterDir ?? null,
      adapterDir: handle.adapterDir ?? handle.dir ?? null,
      mode: handle.mode ?? 'lora',
      modelConfig: handle.modelConfig ?? {},
      meta,
    };
    await writeFileAtomic(file, `${JSON.stringify(payload)}\n`);
    return { file, sha256: await sha256File(file), kind };
  },

  // The spec and command are written to disk even when the backend is
  // unavailable, so the training recipe this backend would run is reviewable
  // without running it.
  async renderTrainingScript({ handle, examplesFile, hyper = {}, budget = {}, outDir, mode, resumeFrom = null }) {
    await ensureDir(outDir);
    const limits = limitsFrom(budget, { ...hyper, maxSeqLen: handle?.modelConfig?.maxSeqLen ?? hyper.maxSeqLen });
    const spec = {
      op: 'train',
      modelId: handle?.modelId ?? 'MODEL_ID',
      adapterDir: handle?.adapterDir ?? null,
      mode: mode ?? handle?.mode ?? 'lora',
      hf: handle?.modelConfig?.hf ?? {},
      examplesFile,
      outDir,
      resumeFrom,
      batchSize: hyper.batchSize ?? 2,
      lr: hyper.lr ?? 1e-3,
      seed: hyper.seed ?? 1,
      lora: hyper.lora ?? { rank: 8, alpha: 16, targets: ['q_proj', 'v_proj'] },
      budget: {
        maxSteps: budget.maxSteps ?? 20,
        maxTokens: budget.maxTokens ?? null,
        maxSeconds: budget.maxSeconds ?? null,
        checkpointEvery: budget.checkpointEvery ?? 0,
      },
      ...limits,
    };
    const specFile = path.join(outDir, 'hf-peft-train-spec.json');
    await writeFileAtomic(specFile, `${JSON.stringify(spec, null, 2)}\n`);
    const file = path.join(outDir, 'hf-peft-train.sh');
    await writeFileAtomic(file, `#!/bin/sh\n# exact command executed by the hf-peft backend\n${pythonBin()} ${RUNNER} ${specFile}\n`);
    return { file, specFile, runner: RUNNER, spec, sha256: await sha256File(file) };
  },

  async train({
    handle,
    examplesFile,
    examples,
    mode,
    hyper = {},
    budget = {},
    resume = null,
    signal = null,
    outDir = path.join(process.cwd(), '.tmf-run', 'hf-peft'),
  } = {}) {
    const probe = await backend.probe();
    // The builtin backend takes in-memory examples; this one needs a file for
    // the Python worker. Accepting both keeps the two backends callable through
    // one parameter contract instead of making every caller branch on backend
    // id. The materialized file is written under the run's own outDir.
    let trainFile = examplesFile;
    if (!trainFile && examples?.length) {
      await ensureDir(outDir);
      trainFile = path.join(outDir, 'train-examples.jsonl');
      await writeFileAtomic(trainFile, `${examples.map((e) => JSON.stringify(e)).join('\n')}\n`);
    }
    // A cancelled CLI job re-enters the train job-step without an explicit
    // backend resume argument. Recover only a complete Trainer checkpoint;
    // anything after that checkpoint is deliberately replayed.
    if (!resume && fs.existsSync(outDir)) {
      const checkpoints = fs.readdirSync(outDir, { withFileTypes: true })
        .filter((entry) => entry.isDirectory() && /^checkpoint-\d+$/.test(entry.name))
        .map((entry) => ({ step: Number(entry.name.slice('checkpoint-'.length)), path: path.join(outDir, entry.name) }))
        .filter((entry) => fs.existsSync(path.join(entry.path, 'trainer_state.json')))
        .sort((a, b) => b.step - a.step);
      resume = checkpoints[0]?.path ?? null;
    }
    const rendered = await backend.renderTrainingScript({
      handle, examplesFile: trainFile, hyper, budget, outDir, mode, resumeFrom: resume,
    });
    if (!probe.available) {
      const err = unavailable('train', probe);
      err.renderedScript = rendered.file;
      err.renderedSpec = rendered.specFile;
      throw err;
    }
    if (!trainFile) throw new Error('hf-peft train() requires examplesFile or examples (prompt/completion records)');
    const { spec } = rendered;
    const timeoutMs = ((budget.maxSeconds ?? 1800) + 120) * 1000;
    const { payload, specFile } = await runOp('train', spec, { outDir, timeoutMs, signal });
    return {
      metrics: payload.metrics,
      history: payload.history,
      stopReason: payload.metrics.stopReason,
      // There is no portable optimizer state to hand back: it lives in the
      // Trainer checkpoint directories, which are what resume consumes.
      optimizerState: null,
      checkpoints: payload.checkpoints,
      handle: { ...handle, mode: spec.mode, dir: payload.adapterDir, adapterDir: payload.adapterDir },
      script: rendered.file,
      specFile,
      limits: payload.limits,
    };
  },

  // Independent evaluation: this loads the base model (plus the adapter when
  // the handle has one) in a fresh process and scores the given examples. It
  // shares no state with train(), so a baseline and a tuned run are two
  // separate measurements rather than two reads of one training log.
  async evaluate({ handle, examplesFile, examples, hyper = {}, budget = {}, signal = null, outDir } = {}) {
    const probe = await backend.probe();
    if (!probe.available) throw unavailable('evaluate', probe);
    if (!handle?.modelId) throw new Error('hf-peft evaluate() requires a handle with modelId');
    const dir = outDir ?? path.join(process.cwd(), '.tmf-run', 'hf-peft-eval');
    await ensureDir(dir);
    let file = examplesFile;
    if (!file) {
      if (!examples?.length) throw new Error('evaluation needs at least one example');
      file = path.join(dir, 'eval-examples.jsonl');
      await writeFileAtomic(file, `${examples.map((e) => JSON.stringify(e)).join('\n')}\n`);
    }
    const spec = {
      modelId: handle.modelId,
      adapterDir: handle.adapterDir ?? handle.dir ?? null,
      mode: handle.mode ?? 'lora',
      hf: handle.modelConfig?.hf ?? {},
      examplesFile: file,
      ...limitsFrom(budget, {
        ...hyper,
        maxMemoryMb: handle.modelConfig?.hf?.maxMemoryMb ?? hyper.maxMemoryMb,
        maxSeqLen: handle.modelConfig?.maxSeqLen ?? hyper.maxSeqLen,
      }),
    };
    const { payload } = await runOp('evaluate', spec, {
      outDir: dir, timeoutMs: ((budget.maxSeconds ?? 600) + 120) * 1000, signal,
    });
    return {
      loss: payload.loss,
      perplexity: payload.perplexity,
      tokens: payload.tokens,
      examples: payload.examples,
      adapterLoaded: payload.adapterLoaded,
      device: payload.device,
      backend: backend.id,
      limits: payload.limits,
    };
  },

  // One offline model load scores both loss splits and generates the holdout
  // continuations. The evaluator keeps the same exact-match diagnostics and
  // gate logic; only the redundant Python/model startups are removed.
  async evaluateSplitSet({
    handle, holdoutExamples, validationExamples = [], prompts = [], maxNewTokens = 48,
    budget = {}, signal = null, outDir,
  } = {}) {
    const probe = await backend.probe();
    if (!probe.available) throw unavailable('evaluate split set', probe);
    if (!handle?.modelId || !Array.isArray(holdoutExamples) || !holdoutExamples.length) {
      throw new Error('hf-peft evaluateSplitSet needs a model and non-empty holdout examples');
    }
    const dir = outDir ?? path.join(process.cwd(), '.tmf-run', 'hf-peft-eval-bundle');
    await ensureDir(dir);
    const holdoutFile = path.join(dir, 'holdout-examples.jsonl');
    const validationFile = path.join(dir, 'validation-examples.jsonl');
    const promptsFile = path.join(dir, 'prompts.jsonl');
    await writeFileAtomic(holdoutFile, `${holdoutExamples.map((e) => JSON.stringify(e)).join('\n')}\n`);
    await writeFileAtomic(validationFile, `${validationExamples.map((e) => JSON.stringify(e)).join('\n')}${validationExamples.length ? '\n' : ''}`);
    await writeFileAtomic(promptsFile, `${prompts.map((prompt) => JSON.stringify({ prompt })).join('\n')}${prompts.length ? '\n' : ''}`);
    const hf = handle.modelConfig?.hf ?? {};
    const spec = {
      modelId: handle.modelId, adapterDir: handle.adapterDir ?? handle.dir ?? null,
      mode: handle.mode ?? 'lora', hf, holdoutFile, validationFile, promptsFile, maxNewTokens,
      ...limitsFrom(budget, {
        maxMemoryMb: hf.maxMemoryMb,
        maxSeqLen: handle.modelConfig?.maxSeqLen,
      }),
    };
    const maxSeconds = budget.maxSeconds ?? hf.evalMaxSeconds ?? 1800;
    const { payload } = await runOp('evaluate_bundle', spec, {
      outDir: dir, timeoutMs: (maxSeconds + 120) * 1000, signal,
    });
    if (!Array.isArray(payload.texts) || payload.texts.length !== prompts.length) {
      throw new Error(`hf-peft evaluate_bundle returned ${payload.texts?.length ?? 'invalid'} predictions for ${prompts.length} prompts`);
    }
    const metrics = (part) => part && ({
      loss: part.loss, perplexity: part.perplexity, tokens: part.tokens,
      examples: part.examples, adapterLoaded: payload.adapterLoaded,
      device: payload.device, backend: backend.id, limits: payload.limits,
      wallClockSeconds: payload.wallClockSeconds, peakRssMb: payload.peakRssMb,
      peakCudaAllocatedMb: payload.peakCudaAllocatedMb,
    });
    return { holdout: metrics(payload.holdout), validation: metrics(payload.validation), texts: payload.texts };
  },

  async generate({
    handle, prompt, maxNewTokens = 16, temperature = 0, seed = 1,
    hyper = {}, budget = {}, signal = null, outDir,
  } = {}) {
    const probe = await backend.probe();
    if (!probe.available) throw unavailable('generate', probe);
    if (!handle?.modelId) throw new Error('hf-peft generate() requires a handle with modelId');
    const dir = outDir ?? path.join(process.cwd(), '.tmf-run', 'hf-peft-gen');
    const spec = {
      modelId: handle.modelId,
      adapterDir: handle.adapterDir ?? handle.dir ?? null,
      mode: handle.mode ?? 'lora',
      hf: handle.modelConfig?.hf ?? {},
      prompt: prompt ?? '',
      maxNewTokens,
      temperature,
      seed,
      ...limitsFrom(budget, hyper),
    };
    const { payload } = await runOp('generate', spec, {
      outDir: dir, timeoutMs: ((budget.maxSeconds ?? 300) + 120) * 1000, signal,
    });
    // Same shape as the builtin backend: callers get the decoded continuation.
    return payload.text;
  },

  async generateBatch({
    handle, prompts, maxNewTokens = 16, temperature = 0, seed = 1,
    hyper = {}, budget = {}, signal = null, outDir,
  } = {}) {
    const probe = await backend.probe();
    if (!probe.available) throw unavailable('generate a batch', probe);
    if (!handle?.modelId) throw new Error('hf-peft generateBatch() requires a handle with modelId');
    if (!Array.isArray(prompts) || prompts.length === 0 || prompts.some((prompt) => typeof prompt !== 'string')) {
      throw new Error('hf-peft generateBatch() requires a non-empty array of string prompts');
    }
    const dir = outDir ?? path.join(process.cwd(), '.tmf-run', 'hf-peft-gen-batch');
    await ensureDir(dir);
    const promptsFile = path.join(dir, 'prompts.jsonl');
    await writeFileAtomic(promptsFile, `${prompts.map((prompt) => JSON.stringify({ prompt })).join('\n')}\n`);
    const spec = {
      modelId: handle.modelId,
      adapterDir: handle.adapterDir ?? handle.dir ?? null,
      mode: handle.mode ?? 'lora',
      hf: handle.modelConfig?.hf ?? {},
      promptsFile,
      maxNewTokens,
      temperature,
      seed,
      ...limitsFrom(budget, hyper),
    };
    const { payload } = await runOp('generate_batch', spec, {
      outDir: dir, timeoutMs: ((budget.maxSeconds ?? 600) + 120) * 1000, signal,
    });
    if (!Array.isArray(payload.texts) || payload.texts.length !== prompts.length) {
      throw new Error(`hf-peft generateBatch returned ${payload.texts?.length ?? 'invalid'} result(s) for ${prompts.length} prompt(s)`);
    }
    return payload.texts;
  },
};

export default backend;
