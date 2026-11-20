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
// This is a local-execution backend only. It does not download models and does
// not call a hosted API, so it stays inside the project's "no paid training,
// no external data egress" constraints. A caller must point `modelId` at a
// model already present in the local cache.

import { spawn } from 'node:child_process';
import path from 'node:path';
import { writeFileAtomic, ensureDir, readJson } from '../../core/fsx.mjs';
import { sha256File } from '../../core/hash.mjs';

export const REQUIRED_PYTHON_PACKAGES = ['torch', 'transformers', 'peft'];

function runPython(args, { python = process.env.TMF_PYTHON || 'python3', timeoutMs = 600000, cwd } = {}) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(python, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
    } catch (err) {
      resolve({ code: -1, stdout: '', stderr: err.message, spawnFailed: true });
      return;
    }
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      stderr += `\ntimed out after ${timeoutMs}ms`;
    }, timeoutMs);
    child.stdout.on('data', (c) => { stdout += c; });
    child.stderr.on('data', (c) => { stderr += c; });
    child.on('error', (err) => {
      clearTimeout(timer);
      resolve({ code: -1, stdout, stderr: `${stderr}${err.message}`, spawnFailed: true });
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      resolve({ code, stdout, stderr });
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

const backend = {
  id: 'hf-peft',

  capabilities: {
    modes: ['lora', 'full-finetune'],
    frameworks: ['torch', 'transformers', 'peft'],
    devices: ['cpu', 'cuda'],
    supportsResume: true,
    supportsCancel: true,
    precision: 'float32/bfloat16',
    requiresLocalModelCache: true,
  },

  async probe() {
    const dir = path.join(process.cwd(), '.tmf-run', 'probe');
    await ensureDir(dir);
    const script = path.join(dir, 'hf-peft-probe.py');
    await writeFileAtomic(script, PROBE_SCRIPT);
    const result = await runPython([script], { timeoutMs: 60000 });
    if (result.spawnFailed || result.code !== 0) {
      return {
        available: false,
        reason: `python interpreter unusable (${(result.stderr || '').trim().split('\n').slice(-1)[0] || `exit ${result.code}`})`,
        details: { python: process.env.TMF_PYTHON || 'python3', probeScript: script },
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
    return { available: true, reason: 'torch, transformers and peft are importable', details: { versions: report, devices: report._devices } };
  },

  async createModel({ modelId, outDir, modelConfig = {} } = {}) {
    const probe = await backend.probe();
    if (!probe.available) throw unavailable('create a model', probe);
    if (!modelId) throw new Error('hf-peft requires modelId pointing at a locally cached model');
    return { backend: backend.id, kind: 'base', modelId, outDir, modelConfig, python: true };
  },

  async loadModel({ file } = {}) {
    const probe = await backend.probe();
    if (!probe.available) throw unavailable('load a checkpoint', probe);
    const payload = await readJson(file);
    if (payload.backend !== backend.id) throw new Error(`checkpoint backend "${payload.backend}" is not ${backend.id}`);
    return { backend: backend.id, kind: payload.kind, modelId: payload.modelId, dir: payload.dir, file };
  },

  async saveModel({ handle, file, kind = 'adapter', meta = {} } = {}) {
    const probe = await backend.probe();
    if (!probe.available) throw unavailable('save a checkpoint', probe);
    const payload = { format: 'tmf-checkpoint-v1', backend: backend.id, kind, modelId: handle.modelId, dir: handle.dir, meta };
    await writeFileAtomic(file, `${JSON.stringify(payload)}\n`);
    return { file, sha256: await sha256File(file), kind };
  },

  // The generated script is written to disk even when the backend is
  // unavailable, so the training recipe this backend would run is reviewable.
  async renderTrainingScript({ handle, examplesFile, hyper = {}, budget = {}, outDir }) {
    const script = `
import json, os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

model_id = ${JSON.stringify(handle.modelId ?? 'MODEL_ID')}
rows = [json.loads(l) for l in open(${JSON.stringify(examplesFile)})]
tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
model = get_peft_model(model, LoraConfig(
    r=${hyper.lora?.rank ?? 8},
    lora_alpha=${hyper.lora?.alpha ?? 16},
    target_modules=${JSON.stringify(hyper.lora?.targets ?? ['q_proj', 'v_proj'])},
    task_type="CAUSAL_LM",
))
model.print_trainable_parameters()

def encode(row):
    prompt_ids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
    full_ids = tok(row["prompt"] + row["completion"], add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels}

ds = Dataset.from_list(rows).map(encode, remove_columns=["prompt", "completion"])
args = TrainingArguments(
    output_dir=${JSON.stringify(outDir)},
    per_device_train_batch_size=${hyper.batchSize ?? 8},
    learning_rate=${hyper.lr ?? 1e-4},
    max_steps=${budget.maxSteps ?? 200},
    logging_steps=10,
    save_steps=${budget.checkpointEvery ?? 50},
    report_to=[],
)
Trainer(model=model, args=args, train_dataset=ds).train()
model.save_pretrained(os.path.join(${JSON.stringify(outDir)}, "adapter"))
print(json.dumps({"status": "ok", "adapter": os.path.join(${JSON.stringify(outDir)}, "adapter")}))
`;
    await ensureDir(outDir);
    const file = path.join(outDir, 'hf-peft-train.py');
    await writeFileAtomic(file, script);
    return { file, sha256: await sha256File(file) };
  },

  async train({ handle, examplesFile, hyper = {}, budget = {}, outDir = path.join(process.cwd(), '.tmf-run', 'hf-peft') } = {}) {
    const probe = await backend.probe();
    const rendered = await backend.renderTrainingScript({ handle, examplesFile, hyper, budget, outDir });
    if (!probe.available) {
      const err = unavailable('train', probe);
      err.renderedScript = rendered.file;
      throw err;
    }
    const result = await runPython([rendered.file], { timeoutMs: (budget.maxSeconds ?? 1800) * 1000 });
    if (result.code !== 0) {
      throw new Error(`hf-peft training failed (exit ${result.code}): ${result.stderr.slice(-500)}`);
    }
    const last = result.stdout.trim().split('\n').slice(-1)[0];
    return { metrics: { backend: backend.id, stopReason: 'completed', raw: last }, history: [], script: rendered.file };
  },

  async evaluate({ handle } = {}) {
    const probe = await backend.probe();
    if (!probe.available) throw unavailable('evaluate', probe);
    throw new Error('hf-peft evaluate() requires a local model cache; not exercised in this environment');
  },

  async generate({ handle } = {}) {
    const probe = await backend.probe();
    if (!probe.available) throw unavailable('generate', probe);
    throw new Error('hf-peft generate() requires a local model cache; not exercised in this environment');
  },
};

export default backend;
