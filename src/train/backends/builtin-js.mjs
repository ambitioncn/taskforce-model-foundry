// The default trainer backend: a real optimizer loop over the pure-JS
// transformer.
//
// "Real" here means the specific things the acceptance plan asks for. Gradients
// come from the autodiff engine and are numerically verified in the test suite;
// AdamW mutates the parameter arrays in place; the resulting checkpoint is a
// byte-for-byte record of those arrays and reloads to identical outputs. No
// retrieval, no prompt scaffolding and no recorded numbers stand in for the
// update.

import path from 'node:path';
import { backward } from '../../core/autograd.mjs';
import { sha256File, sha256Json } from '../../core/hash.mjs';
import { writeJson, readJson, ensureDir } from '../../core/fsx.mjs';
import { Rng } from '../../core/rng.mjs';
import { TransformerModel, snapshotParams, l2Delta, DEFAULT_MODEL_CONFIG, DEFAULT_LORA_CONFIG } from '../../model/transformer.mjs';
import { Tokenizer } from '../../model/tokenizer.mjs';
import { AdamW, lrSchedule, DEFAULT_OPTIMIZER_CONFIG } from '../optimizer.mjs';
import { makeBatches, countTokens, encodeRecords } from '../dataset.mjs';

export const CHECKPOINT_FORMAT = 'tmf-checkpoint-v1';

export const DEFAULT_HYPER = {
  ...DEFAULT_OPTIMIZER_CONFIG,
  batchSize: 8,
  warmupRatio: 0.1,
  minLrRatio: 0.1,
};

export const DEFAULT_BUDGET = {
  maxSteps: 200,
  maxSeconds: 120,
  maxTokens: null,
  targetLoss: null,
};

// `hyper.steps` is the knob `tmf init` writes, so a config that sets it must
// actually get that many steps. The budget object stays authoritative when it
// names maxSteps itself; otherwise hyper.steps supplies it. Without this the
// step count in the config was silently ignored in favour of DEFAULT_BUDGET.
export function resolveStepBudget({ budget = {}, hyper = {} } = {}) {
  const fromBudget = budget?.maxSteps;
  if (fromBudget !== undefined) return { maxSteps: fromBudget, source: 'trainBudget.maxSteps' };
  const fromHyper = hyper?.steps;
  if (fromHyper !== undefined && fromHyper !== null) {
    if (!Number.isFinite(fromHyper) || fromHyper <= 0) {
      throw new Error(`hyper.steps must be a positive number or null, got ${fromHyper}`);
    }
    return { maxSteps: fromHyper, source: 'hyper.steps' };
  }
  return { maxSteps: DEFAULT_BUDGET.maxSteps, source: 'default' };
}

function resolveBudget(budget = {}, { hyper = {} } = {}) {
  const step = resolveStepBudget({ budget, hyper });
  const resolved = { ...DEFAULT_BUDGET, ...budget, maxSteps: step.maxSteps };
  resolved.stepBudgetSource = step.source;
  for (const key of ['maxSteps', 'maxSeconds', 'maxTokens']) {
    const value = resolved[key];
    if (value !== null && value !== undefined && (!Number.isFinite(value) || value <= 0)) {
      throw new Error(`budget.${key} must be a positive number or null, got ${value}`);
    }
  }
  if (resolved.maxSteps === null && resolved.maxSeconds === null && resolved.maxTokens === null) {
    throw new Error('a training budget needs at least one of maxSteps, maxSeconds or maxTokens');
  }
  return resolved;
}

const backend = {
  id: 'builtin-js',

  capabilities: {
    // Every non-lora mode trains the full parameter set through the same
    // optimizer loop (see trainableParams below), so full-finetune is a real
    // capability here, not an alias we reject at the config boundary.
    modes: ['pretrain', 'lora', 'full-finetune'],
    // This backend owns no tokenizer of its own: the training job builds the
    // vocabulary from the data split and hands it in. See prepareDataset().
    ownsTokenizer: false,
    frameworks: ['builtin-autograd'],
    devices: ['cpu'],
    supportsResume: true,
    supportsCancel: true,
    precision: 'float32',
  },

  // Always available: the implementation ships with the package and has no
  // native dependency, which is why it is the default.
  async probe() {
    return {
      available: true,
      reason: 'bundled pure-JavaScript trainer; no external dependency',
      details: { node: process.version, deps: [] },
    };
  },

  // Dataset preparation is part of the backend contract so the training job
  // does not have to know whether ids come from a job-built vocabulary (here)
  // or from the model's own tokenizer (hf-peft). Both return the same shape.
  prepareDataset({ handle, records, maxSeqLen = 96 } = {}) {
    if (!handle?.tokenizer) throw new Error('builtin-js prepareDataset() requires a handle with a tokenizer');
    const { examples, skipped } = encodeRecords(records ?? [], handle.tokenizer, { maxSeqLen });
    return { examples, skipped, tokens: countTokens(examples), tokenizer: 'job-built', vocabSize: handle.tokenizer.size };
  },

  async createModel({ modelConfig = {}, tokenizer, seed = 1 } = {}) {
    if (!tokenizer) throw new Error('createModel requires a tokenizer');
    const config = { ...DEFAULT_MODEL_CONFIG, ...modelConfig, vocabSize: tokenizer.size };
    const model = new TransformerModel(config, { seed });
    return { backend: backend.id, kind: 'base', model, tokenizer, seed, modelConfig: config };
  },

  async loadModel({ file, attachAdapter = null } = {}) {
    const payload = await readJson(file);
    if (payload.format !== CHECKPOINT_FORMAT) {
      throw new Error(`unsupported checkpoint format "${payload.format}"; expected ${CHECKPOINT_FORMAT}`);
    }
    if (payload.backend !== backend.id) {
      throw new Error(`checkpoint was produced by backend "${payload.backend}", not ${backend.id}`);
    }
    const tokenizer = Tokenizer.fromJSON(payload.tokenizer);
    const model = new TransformerModel(payload.modelConfig, { seed: payload.seed });
    if (payload.kind === 'base') {
      model.loadStateDict(payload.params);
    } else if (payload.kind === 'adapter') {
      if (!payload.baseCheckpoint) throw new Error('adapter checkpoint does not record its base checkpoint');
      throw new Error('load an adapter with loadAdapter(), which needs the base checkpoint first');
    } else {
      // Falling through would hand back freshly initialised weights and
      // silently evaluate the wrong model, so an unknown kind is terminal.
      throw new Error(`unsupported checkpoint kind "${payload.kind}"; expected base or adapter`);
    }
    const handle = { backend: backend.id, kind: 'base', model, tokenizer, seed: payload.seed, modelConfig: payload.modelConfig, meta: payload.meta ?? {} };
    if (attachAdapter) await backend.loadAdapter(handle, attachAdapter);
    return handle;
  },

  // Adapters are layered onto an already loaded base so that "reload the
  // adapter and get identical outputs" is directly testable.
  async loadAdapter(handle, file) {
    const payload = await readJson(file);
    if (payload.format !== CHECKPOINT_FORMAT || payload.kind !== 'adapter') {
      throw new Error(`"${file}" is not a ${CHECKPOINT_FORMAT} adapter checkpoint`);
    }
    if (payload.baseFingerprint && payload.baseFingerprint !== backend.fingerprint(handle)) {
      throw new Error(
        'adapter does not match this base model: fingerprint '
        + `${payload.baseFingerprint.slice(0, 12)} != ${backend.fingerprint(handle).slice(0, 12)}`,
      );
    }
    handle.model.loadAdapterState(payload.params);
    handle.adapterMeta = payload.meta ?? {};
    handle.kind = 'base+adapter';
    return handle;
  },

  // Digest of the base weights; used to bind an adapter to the base it was
  // trained against and to prove in evidence that weights changed.
  fingerprint(handle) {
    return sha256Json(handle.model.stateDict());
  },

  async saveModel({ handle, file, kind = 'base', meta = {} } = {}) {
    await ensureDir(path.dirname(file));
    const payload = {
      format: CHECKPOINT_FORMAT,
      backend: backend.id,
      kind,
      createdAt: new Date().toISOString(),
      seed: handle.seed,
      modelConfig: handle.modelConfig,
      tokenizer: handle.tokenizer.toJSON(),
      loraConfig: kind === 'adapter' ? handle.model.loraConfig : null,
      baseFingerprint: kind === 'adapter' ? (meta.baseFingerprint ?? null) : null,
      baseCheckpoint: kind === 'adapter' ? (meta.baseCheckpoint ?? null) : null,
      params: kind === 'adapter' ? handle.model.adapterState() : handle.model.stateDict(),
      meta,
    };
    await writeJson(file, payload, { pretty: false });
    return { file, sha256: await sha256File(file), kind, bytes: JSON.stringify(payload).length };
  },

  async train({ handle, examples, mode = 'pretrain', hyper = {}, budget = {}, hooks = {}, seed = 'tmf-train', resume = null } = {}) {
    if (!examples?.length) throw new Error('training needs at least one example');
    const cfg = { ...DEFAULT_HYPER, ...hyper };
    const limits = resolveBudget(budget, { hyper: cfg });
    const model = handle.model;

    if (mode === 'lora') {
      if (!model.loraAttached) model.attachLora(cfg.lora ?? DEFAULT_LORA_CONFIG);
      model.setBaseFrozen(true);
    } else {
      model.setBaseFrozen(false);
    }
    const params = model.trainableParams({ mode: mode === 'lora' ? 'lora' : 'full' });
    const optimizer = new AdamW(params, cfg);
    if (resume?.optimizer) optimizer.loadStateDict(resume.optimizer);

    const before = snapshotParams(params);
    const startStep = resume?.step ?? 0;
    const totalSteps = limits.maxSteps ?? 0;
    const warmupSteps = Math.max(0, Math.floor((totalSteps || 0) * cfg.warmupRatio));
    const startedAt = Date.now();

    const history = [];
    let step = startStep;
    let tokensSeen = resume?.tokensSeen ?? 0;
    let stopReason = null;
    let lastLoss = null;
    let epoch = resume?.epoch ?? 0;
    // Batch loss is noisy and the tail of training diverged (final >> best), so
    // track an EMA-smoothed best and keep a periodic snapshot to restore from.
    const emaAlpha = 0.1;
    const snapshotEvery = 10;
    let emaLoss = null;
    let bestEmaLoss = null;
    let bestSnapshot = null;
    let bestSnapshotStep = null;

    outer: while (true) {
      const rng = new Rng(`${seed}:epoch:${epoch}`);
      const batches = makeBatches(examples, cfg.batchSize, rng);
      epoch += 1;
      for (const batch of batches) {
        if (limits.maxSteps !== null && step >= limits.maxSteps) { stopReason = 'max_steps'; break outer; }
        const elapsed = (Date.now() - startedAt) / 1000;
        if (limits.maxSeconds !== null && elapsed >= limits.maxSeconds) { stopReason = 'max_seconds'; break outer; }
        if (limits.maxTokens !== null && tokensSeen >= limits.maxTokens) { stopReason = 'max_tokens'; break outer; }
        if (hooks.shouldCancel && (await hooks.shouldCancel({ step, elapsed }))) { stopReason = 'cancelled'; break outer; }

        optimizer.zeroGrad();
        let batchLoss = 0;
        let batchTokens = 0;
        for (const example of batch) {
          const loss = model.loss(example.ids, { promptLen: example.promptLen });
          backward(loss);
          batchLoss += loss.lossValue;
          batchTokens += loss.tokenCount;
        }
        // Gradients accumulated across the batch; scale to the mean.
        const scaleFactor = 1 / batch.length;
        for (const p of params) if (p.grad) for (let i = 0; i < p.grad.length; i += 1) p.grad[i] *= scaleFactor;

        const lr = cfg.lr * lrSchedule(step, totalSteps || batches.length, { warmupSteps, minRatio: cfg.minLrRatio });
        const applied = optimizer.applyStep({ lr });
        lastLoss = batchLoss / batch.length;
        tokensSeen += batchTokens;
        step += 1;

        const entry = {
          step,
          loss: Number(lastLoss.toFixed(6)),
          lr: Number(lr.toFixed(8)),
          gradNorm: Number(applied.gradNorm.toFixed(6)),
          tokens: tokensSeen,
          elapsedMs: Date.now() - startedAt,
        };
        history.push(entry);
        emaLoss = emaLoss === null ? lastLoss : emaLoss + emaAlpha * (lastLoss - emaLoss);
        if (step > warmupSteps && step % snapshotEvery === 0 && (bestEmaLoss === null || emaLoss < bestEmaLoss)) {
          bestEmaLoss = emaLoss;
          bestSnapshot = snapshotParams(params);
          bestSnapshotStep = step;
        }
        await hooks.onStep?.(entry, { optimizer, model, handle });

        if (limits.targetLoss !== null && limits.targetLoss !== undefined && lastLoss <= limits.targetLoss) {
          stopReason = 'target_loss';
          break outer;
        }
      }
      if (examples.length === 0) break;
    }

    // Finalize on the best smoothed weights instead of whatever the last step left.
    let restoredFromStep = null;
    if (bestSnapshot && emaLoss !== null && bestEmaLoss !== null && emaLoss > bestEmaLoss) {
      for (const p of params) {
        const saved = bestSnapshot[p.name];
        if (saved) p.data.set(saved);
      }
      restoredFromStep = bestSnapshotStep;
    }

    const after = snapshotParams(params);
    const delta = l2Delta(before, after);
    const metrics = {
      mode,
      steps: step - startStep,
      totalSteps: step,
      epochs: epoch,
      firstLoss: history.length ? history[0].loss : null,
      finalLoss: lastLoss === null ? null : Number(lastLoss.toFixed(6)),
      bestLoss: history.length ? Math.min(...history.map((h) => h.loss)) : null,
      finalSmoothedLoss: emaLoss === null ? null : Number(emaLoss.toFixed(6)),
      bestSmoothedLoss: bestEmaLoss === null ? null : Number(bestEmaLoss.toFixed(6)),
      restoredFromStep,
      tokensSeen,
      trainableParams: params.reduce((n, p) => n + p.size, 0),
      // Non-zero L2 displacement is the direct evidence that weights moved.
      parameterL2Delta: Number(delta.toFixed(8)),
      wallClockSeconds: Number(((Date.now() - startedAt) / 1000).toFixed(3)),
      stepBudget: limits.maxSteps,
      stepBudgetSource: limits.stepBudgetSource,
      stopReason: stopReason ?? 'completed',
    };
    return { metrics, history, optimizerState: optimizer.stateDict(), stopReason: metrics.stopReason };
  },

  async evaluate({ handle, examples } = {}) {
    if (!examples?.length) throw new Error('evaluation needs at least one example');
    let total = 0;
    let tokens = 0;
    for (const example of examples) {
      const loss = handle.model.loss(example.ids, { promptLen: example.promptLen });
      total += loss.lossValue * loss.tokenCount;
      tokens += loss.tokenCount;
    }
    const loss = tokens > 0 ? total / tokens : 0;
    return { loss: Number(loss.toFixed(6)), perplexity: Number(Math.exp(loss).toFixed(6)), tokens, examples: examples.length };
  },

  async generate({ handle, prompt, maxNewTokens = 32, temperature = 0, stopToken = null, seed = 'tmf-gen' } = {}) {
    const { model, tokenizer } = handle;
    const rng = new Rng(seed);
    let ids = tokenizer.encode(prompt, { bos: true });
    const stopId = stopToken === null ? tokenizer.eosId : tokenizer.index.get(stopToken) ?? tokenizer.eosId;
    const generated = [];
    for (let i = 0; i < maxNewTokens; i += 1) {
      if (ids.length >= model.config.maxSeqLen) break;
      const logits = model.forward(ids);
      const row = logits.rows - 1;
      let nextId;
      if (temperature <= 0) {
        // Greedy decoding keeps generation reproducible for evaluation.
        let best = -Infinity;
        nextId = 0;
        for (let c = 0; c < logits.cols; c += 1) {
          const v = logits.at(row, c);
          if (v > best) { best = v; nextId = c; }
        }
      } else {
        let max = -Infinity;
        for (let c = 0; c < logits.cols; c += 1) max = Math.max(max, logits.at(row, c) / temperature);
        let sum = 0;
        const probs = new Float64Array(logits.cols);
        for (let c = 0; c < logits.cols; c += 1) {
          probs[c] = Math.exp(logits.at(row, c) / temperature - max);
          sum += probs[c];
        }
        let draw = rng.next() * sum;
        nextId = logits.cols - 1;
        for (let c = 0; c < logits.cols; c += 1) {
          draw -= probs[c];
          if (draw <= 0) { nextId = c; break; }
        }
      }
      if (nextId === stopId) break;
      generated.push(nextId);
      ids = [...ids, nextId];
    }
    return tokenizer.decode(generated);
  },
};

export default backend;
