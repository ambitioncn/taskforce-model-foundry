#!/usr/bin/env node
// The `tmf` command line.
//
// A thin, honest shell over src/api.mjs: it parses flags, prints human or JSON
// output, and maps outcomes to exit codes (0 ok, 1 failed check/invalid input,
// 2 usage error). It contains no training logic of its own.

import path from 'node:path';
import process from 'node:process';
import { doctor, synthesize, train, status, evaluate, exportModel, screenplayQualityPrepare, screenplayQualityFinalize } from '../src/api.mjs';
import {
  loadConfig, validateConfig, exampleConfig, formatConfigErrors, ConfigError,
  DEFAULT_CONFIG_FILES,
} from '../src/config.mjs';
import { writeJson, readJson, exists } from '../src/core/fsx.mjs';
import { VERSION } from '../src/index.mjs';
import { startBackgroundEvaluation, backgroundEvaluationStatus } from '../src/integrations/async-evaluate.mjs';
import { startBackgroundTraining, backgroundTrainingStatus, cancelBackgroundTraining } from '../src/integrations/async-train.mjs';
import { createScreenplayQualityContract } from '../src/eval/screenplay-quality.mjs';

const USAGE = `taskforce-model-foundry ${VERSION}

Usage: tmf <command> [options]

Commands:
  doctor                 Check environment, backends and config
  init                   Write a starter tmf.config.json (+ synthetic data)
  config validate [file] Validate a config file and print every problem
  backends               List trainer backends and availability
  synth                  Generate a synthetic corpus (no private data needed)
  train                  Run a real post-training job
  evaluate               Score a finished job on an independent holdout
  train-status <job> <requestId>     Read a background training receipt
  evaluate-status <job> <requestId>  Read a background evaluation receipt
  train-cancel <job> <requestId>     Cancel background training
  quality-contract       Freeze the 120-question screenplay gate before a new job
  quality-prepare        Score full screenplay suite and create blinded DAPT pairs
  quality-finalize       Tally two judges and clean-val loss; seal external report
  export                 Package a deliverable bundle (gated on evaluation)
                         --report <file> reuses an existing eval report instead of re-evaluating
                         --quality-report <file> pins predeclared external quality evidence
  status <jobId>         Show durable job state and recent audit events
  version                Print the version
  help                   Print this help

Options:
  -c, --config <file>    Config file (default: ${DEFAULT_CONFIG_FILES.join(' or ')})
  -C, --cwd <dir>        Base directory (default: current directory)
      --json             Machine-readable JSON output
      --job <id>         Job id (train)
      --out <dir>        Output directory (synth/init, default: data)
      --count <n>        Record count (synth/init, default: 160)
      --resume-cancelled Resume a job that an operator cancelled
      --no-resume        Ignore existing job state instead of resuming
      --background       Run train/evaluate in an independent worker; return requestId
      --retry            Explicitly retry a failed/interrupted background request

Exit codes:
  0 ok   1 the job ran and did not succeed   2 usage error
  130 train was cancelled by SIGINT/SIGTERM (resume with --resume-cancelled)
`;

function parseArgs(argv) {
  const opts = { _: [], json: false, resume: true, resumeCancelled: false, background: false, retry: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      const value = argv[i + 1];
      if (value === undefined || value.startsWith('-')) throw new UsageError(`${arg} requires a value`);
      i += 1;
      return value;
    };
    switch (arg) {
      case '-c': case '--config': opts.config = next(); break;
      case '-C': case '--cwd': opts.cwd = next(); break;
      case '--json': opts.json = true; break;
      case '--job': opts.job = next(); break;
      case '--out': opts.out = next(); break;
      case '--report': opts.report = next(); break;
      case '--suite': opts.suite = next(); break;
      case '--scripts-dir': opts.scriptsDir = next(); break;
      case '--base-outputs': opts.baseOutputs = next(); break;
      case '--dapt-outputs': opts.daptOutputs = next(); break;
      case '--candidate-outputs': opts.candidateOutputs = next(); break;
      case '--clean-dapt-report': opts.cleanDaptReport = next(); break;
      case '--clean-dapt-model': opts.cleanDaptModel = next(); break;
      case '--stage': opts.stage = next(); break;
      case '--judge': (opts.judgeVerdicts ??= []).push(next()); break;
      case '--clean-candidate-report': opts.cleanCandidateReport = next(); break;
      case '--clean-candidate-model': opts.cleanCandidateModel = next(); break;
      case '--quality-report': opts.qualityReport = next(); break;
      case '--bundle-version': opts.bundleVersion = next(); break;
      case '--count': opts.count = Number(next()); break;
      case '--resume-cancelled': opts.resumeCancelled = true; break;
      case '--no-resume': opts.resume = false; break;
      case '--background': opts.background = true; break;
      case '--retry': opts.retry = true; break;
      case '-h': case '--help': opts.help = true; break;
      case '-v': case '--version': opts.version = true; break;
      default:
        if (arg.startsWith('-')) throw new UsageError(`unknown option: ${arg}`);
        opts._.push(arg);
    }
  }
  return opts;
}

class UsageError extends Error {}

// Runner step records are objects; older state files may hold bare names.
function stepNames(steps) {
  return (steps ?? []).map((step) => (typeof step === 'string' ? step : step?.name)).filter(Boolean);
}

function out(opts, payload, lines) {
  if (opts.json) console.log(JSON.stringify(payload, null, 2));
  else for (const line of lines) console.log(line);
}

async function cmdDoctor(opts, baseDir) {
  const report = await doctor({ baseDir, configFile: opts.config ?? null, requireConfig: false });
  out(opts, report, [
    `tmf doctor — ${report.ok ? 'OK' : 'FAILED'}`,
    `node ${report.node} on ${report.platform}`,
    ...report.checks.map((c) => `  [${c.ok ? 'ok ' : c.level === 'required' ? 'FAIL' : 'warn'}] ${c.id}: ${c.detail}`),
    ...(report.configErrors.length ? ['config problems:', formatConfigErrors(report.configErrors)] : []),
  ]);
  return report.ok ? 0 : 1;
}

async function cmdInit(opts, baseDir) {
  const file = path.resolve(baseDir, opts.config ?? DEFAULT_CONFIG_FILES[0]);
  if (await exists(file)) {
    console.error(`refusing to overwrite existing config: ${file}`);
    return 1;
  }
  const dataDir = opts.out ?? 'data';
  const config = exampleConfig({ project: path.basename(baseDir), dataDir });
  const corpus = await synthesize({ baseDir, outDir: dataDir, taskCount: opts.count ?? 160 });
  config.data.datasets = Object.entries(corpus.files ?? {}).map(([name, p]) => ({
    name,
    path: path.relative(baseDir, p),
    format: 'jsonl',
    license: 'synthetic',
    source: 'tmf synth',
  }));
  if (config.data.datasets.length === 0) config.data.datasets = exampleConfig({ dataDir }).data.datasets;
  await writeJson(file, config);
  out(opts, { configFile: file, datasets: config.data.datasets }, [
    `wrote ${file}`,
    `synthetic corpus in ${corpus.outDir}`,
    'next: tmf doctor, then tmf train',
  ]);
  return 0;
}

async function cmdConfig(opts, baseDir, rest) {
  if (rest[0] !== 'validate') throw new UsageError('usage: tmf config validate [file]');
  const file = rest[1] ?? opts.config ?? null;
  try {
    const loaded = await loadConfig({ file, baseDir });
    out(opts, { ok: true, file: loaded.file, config: loaded.config }, [`config OK: ${loaded.file}`]);
    return 0;
  } catch (err) {
    if (!(err instanceof ConfigError)) throw err;
    out(opts, { ok: false, errors: err.errors, message: err.message }, [err.message, formatConfigErrors(err.errors)]);
    return 1;
  }
}

async function cmdBackends(opts) {
  const report = await doctor({ baseDir: process.cwd() });
  out(opts, report.backends, report.backends.map(
    (b) => `${b.available ? 'available  ' : 'unavailable'} ${b.id} modes=${(b.capabilities?.modes ?? []).join('|')}${b.available ? '' : ` (${b.reason})`}`,
  ));
  return report.backends.some((b) => b.available) ? 0 : 1;
}

async function cmdSynth(opts, baseDir) {
  const corpus = await synthesize({ baseDir, outDir: opts.out ?? 'data', taskCount: opts.count ?? 160 });
  out(opts, corpus, [`synthetic corpus written to ${corpus.outDir}`,
    ...Object.entries(corpus.files ?? {}).map(([k, v]) => `  ${k}: ${v}`)]);
  return 0;
}

// Operator cancellation from the shell. Node's default SIGINT handling would
// tear the process down mid-step and leave the job record saying `running`
// forever, so `train` installs its own handlers and turns the signal into the
// cooperative abort the runner already understands: the current step finishes
// or throws, state is persisted as `cancelled`, and the written checkpoints
// stay valid input for `--resume-cancelled`. A second signal is taken as "I
// meant now" and exits immediately, accepting the loss of the in-flight step.
function installCancelHandlers(controller) {
  const signals = ['SIGINT', 'SIGTERM'];
  let armed = true;
  const handlers = signals.map((name) => {
    const handler = () => {
      if (!armed) {
        console.error(`\n${name} again: exiting now, the in-flight step is lost`);
        process.exit(130);
      }
      armed = false;
      console.error(`\n${name} received: cancelling after the current step, checkpoints are kept`);
      controller.abort(new Error(`operator cancelled with ${name}`));
    };
    process.on(name, handler);
    return [name, handler];
  });
  return () => { for (const [name, handler] of handlers) process.off(name, handler); };
}

async function cmdTrain(opts, baseDir) {
  const { file, config } = await loadConfig({ file: opts.config ?? null, baseDir });
  if (opts.background) {
    if (!opts.job) throw new UsageError('background training requires --job <stable-id>');
    if (!opts.resume) throw new UsageError('background training does not accept --no-resume');
    const receipt = await startBackgroundTraining({
      baseDir, configFile: file, jobId: opts.job, resumeCancelled: opts.resumeCancelled, retry: opts.retry,
    });
    out(opts, receipt, [`training request ${receipt.requestId}: ${receipt.state}`, `status: tmf train-status ${opts.job} ${receipt.requestId} --config ${file}`, `log: ${receipt.logFile ?? '(pending)'}`]);
    return receipt.state === 'failed' ? 1 : 0;
  }
  if (opts.retry) throw new UsageError('--retry requires --background');
  const controller = new AbortController();
  const disarm = installCancelHandlers(controller);
  let job; let artifacts;
  try {
    ({ job, artifacts } = await train({
      config, baseDir, jobId: opts.job ?? null, resume: opts.resume, resumeCancelled: opts.resumeCancelled,
      signal: controller.signal,
    }));
  } catch (err) {
    // The runner persists `cancelled` and then rethrows, so a cancellation
    // arrives here as an exception even though the job record is intact. A
    // stack trace would read like a crash; report the durable state instead.
    if (err?.name !== 'JobCancelledError' && err?.code !== 'JOB_CANCELLED') throw err;
    const jobId = opts.job ?? `train-${config.project}`;
    const record = await status({ config, baseDir, jobId });
    out(opts, { configFile: file, job: record.state, cancelled: true }, [
      `job ${jobId}: ${record.state?.state ?? 'cancelled'}`,
      `steps: ${stepNames(record.state?.completedSteps).join(' -> ') || '(none)'}`,
      `cancelled: ${err.message}`,
      `resume with: tmf train --config ${file} --job ${jobId} --resume-cancelled`,
    ]);
    return 130;
  } finally {
    disarm();
  }
  const ok = job.state === 'succeeded';
  const cancelled = job.state === 'cancelled';
  out(opts, { configFile: file, job, artifacts }, [
    `job ${job.jobId}: ${job.state}`,
    `steps: ${stepNames(job.completedSteps).join(' -> ') || '(none)'}`,
    ...(ok ? [
      `base checkpoint: ${artifacts.baseCheckpoint}`,
      `${artifacts.adapterCheckpoint ? 'adapter' : 'tuned'} checkpoint: ${artifacts.adapterCheckpoint ?? artifacts.tunedCheckpoint}`,
      `training steps: ${artifacts.metrics?.steps ?? '?'}  parameter L2 delta: ${artifacts.metrics?.parameterL2Delta ?? '?'}`,
    ] : cancelled ? [
      `cancelled: ${job.error?.message ?? 'operator cancelled'}`,
      `resume with: tmf train --config ${file} --job ${job.jobId} --resume-cancelled`,
    ] : [`error: ${job.error?.message ?? 'see audit log'}`]),
  ]);
  // A cancelled job is not a failed check: 130 keeps it distinguishable from
  // the exit 1 that means "the job ran and did not succeed".
  if (cancelled) return 130;
  return ok ? 0 : 1;
}

async function cmdEvaluate(opts, baseDir) {
  const { file, config } = await loadConfig({ file: opts.config ?? null, baseDir });
  if (opts.background) {
    if (!opts.job) throw new UsageError('background evaluation requires --job <stable-id>');
    const receipt = await startBackgroundEvaluation({
      baseDir, configFile: file, jobId: opts.job, reportFile: opts.report ?? null, retry: opts.retry,
    });
    out(opts, receipt, [`evaluation request ${receipt.requestId}: ${receipt.state}`, `status: tmf evaluate-status ${opts.job} ${receipt.requestId} --config ${file}`]);
    return receipt.state === 'failed' ? 1 : 0;
  }
  if (opts.retry) throw new UsageError('--retry requires --background');
  const { artifacts, report } = await evaluate({
    config, baseDir, jobId: opts.job ?? null, reportFile: opts.report ?? null,
  });
  const ok = report.verdict.status === 'passed';
  out(opts, { configFile: file, artifacts, report }, [
    `evaluation ${report.verdict.status}: ${report.verdict.summary}`,
    `holdout records: ${report.inputs?.splits?.holdout ?? '?'}  leakage: ${report.independence?.leaked?.length ?? 0}`,
    ...report.gates.results.map((g) => `  [${g.ok ? 'ok ' : 'FAIL'}] ${g.id}: ${gateLine(g)}`),
    `report: ${report.reportFile ?? '(not written)'}`,
  ]);
  return ok ? 0 : 1;
}

// Gate results carry `reason` only when they fail; a passing gate still needs a
// readable line, so restate the metric it cleared.
function gateLine(g) {
  if (g.reason) return g.reason;
  const comparator = { gte: '>=', lte: '<=', gt: '>', lt: '<' }[g.comparator] ?? g.comparator;
  if (g.actual === null || g.actual === undefined) return `${g.metric} not measurable (needs ${comparator} ${g.threshold})`;
  return `${g.metric}=${g.actual} meets ${comparator} ${g.threshold}`;
}

async function cmdExport(opts, baseDir) {
  const { file, config } = await loadConfig({ file: opts.config ?? null, baseDir });
  let bundle;
  let report;
  // --report pins the bundle to an existing report instead of re-evaluating,
  // so the exported lineage carries the digest that was actually reviewed.
  let evalReport = null;
  if (opts.report) {
    const reportFile = path.resolve(baseDir, opts.report);
    if (!(await exists(reportFile))) throw new UsageError(`eval report not found: ${reportFile}`);
    evalReport = await readJson(reportFile);
  }
  try {
    ({ bundle, report } = await exportModel({
      config, baseDir, jobId: opts.job ?? null, outDir: opts.out ?? null, version: opts.bundleVersion ?? null,
      evalReport, evaluateIfMissing: !evalReport,
      qualityReport: opts.qualityReport ?? null,
    }));
  } catch (err) {
    // A blocked export is an expected outcome, not a crash: report the reason
    // and the failed gates instead of a stack trace.
    if (err?.code !== 'EXPORT_BLOCKED') throw err;
    out(opts, { ok: false, code: err.code, reason: err.message, verdict: err.report?.verdict ?? null }, [
      err.message,
      ...(err.report?.gates?.results ?? []).filter((g) => !g.ok).map((g) => `  [FAIL] ${g.id}: ${gateLine(g)}`),
    ]);
    return 1;
  }
  out(opts, { configFile: file, bundle, verdict: report.verdict }, [
    `exported ${bundle.manifest.name} ${bundle.manifest.version} (${bundle.manifest.kind})`,
    `bundle: ${bundle.outDir}`,
    `digest: ${bundle.bundleDigest}`,
    `files: ${bundle.files.length}  verify: sha256sum -c checksums.txt`,
  ]);
  return 0;
}

async function cmdQualityPrepare(opts, baseDir) {
  const required = ['job', 'report', 'suite', 'baseOutputs', 'daptOutputs', 'candidateOutputs', 'cleanDaptReport', 'cleanDaptModel', 'out'];
  for (const key of required) if (!opts[key]) throw new UsageError(`quality-prepare requires --${key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`);
  const result = await screenplayQualityPrepare({
    baseDir, configFile: opts.config ?? null, jobId: opts.job, evalReport: opts.report,
    suiteFile: opts.suite, scriptsDir: opts.scriptsDir, baseOutputs: opts.baseOutputs,
    daptOutputs: opts.daptOutputs, candidateOutputs: opts.candidateOutputs,
    cleanDaptReport: opts.cleanDaptReport, cleanDaptModel: opts.cleanDaptModel,
    outDir: opts.out,
  });
  out(opts, result, [`quality stage: ${result.stageFile}`, `blind pairs: ${result.blindDir}`]);
  return 0;
}

async function cmdQualityContract(opts) {
  for (const key of ['suite', 'baseOutputs', 'daptOutputs', 'cleanDaptReport']) {
    if (!opts[key]) throw new UsageError(`quality-contract requires --${key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`);
  }
  const contract = await createScreenplayQualityContract({
    suiteFile: opts.suite, baseOutputs: opts.baseOutputs,
    daptOutputs: opts.daptOutputs, cleanDaptReport: opts.cleanDaptReport,
    scriptsDir: opts.scriptsDir,
  });
  if (opts.out) await writeJson(path.resolve(opts.out), contract);
  out(opts, contract, [`frozen screenplay quality contract${opts.out ? `: ${path.resolve(opts.out)}` : ''}`, 'add this object as eval.externalQuality before training']);
  return 0;
}

async function cmdQualityFinalize(opts, baseDir) {
  for (const key of ['stage', 'cleanCandidateReport', 'cleanCandidateModel', 'out']) {
    if (!opts[key]) throw new UsageError(`quality-finalize requires --${key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`);
  }
  if ((opts.judgeVerdicts ?? []).length < 2) throw new UsageError('quality-finalize requires two or more --judge NAME=FILE arguments');
  const result = await screenplayQualityFinalize({
    baseDir, configFile: opts.config ?? null, stageFile: opts.stage,
    judgeVerdicts: opts.judgeVerdicts, cleanCandidateReport: opts.cleanCandidateReport,
    cleanCandidateModel: opts.cleanCandidateModel, reportFile: opts.out,
  });
  out(opts, result, [`quality report: ${result.reportFile}`, `gates: ${result.report.gates.summary}`]);
  return result.report.gates.passed ? 0 : 1;
}

async function cmdStatus(opts, baseDir, rest) {
  const jobId = rest[0] ?? opts.job;
  if (!jobId) throw new UsageError('usage: tmf status <jobId>');
  const { config } = await loadConfig({ file: opts.config ?? null, baseDir });
  const report = await status({ config, baseDir, jobId });
  if (!report.found) {
    out(opts, report, [`job ${jobId}: not found`]);
    return 1;
  }
  out(opts, report, [
    `job ${jobId}: ${report.state.state} (attempt ${report.state.attempt ?? 1})`,
    `steps: ${stepNames(report.state.completedSteps).join(' -> ') || '(none)'}`,
    `audit events: ${report.auditCount}`,
    ...report.audit.map((e) => `  ${e.at ?? ''} ${e.event}`),
  ]);
  return 0;
}

async function cmdBackgroundStatus(opts, baseDir, rest, kind) {
  const [jobId, requestId] = rest;
  if (!jobId || !requestId) throw new UsageError(`usage: tmf ${kind}-status <jobId> <requestId>`);
  const receipt = kind === 'train'
    ? await backgroundTrainingStatus({ baseDir, configFile: opts.config ?? null, jobId, requestId })
    : await backgroundEvaluationStatus({ baseDir, configFile: opts.config ?? null, jobId, requestId });
  out(opts, receipt, [`${kind} request ${requestId}: ${receipt.state}`, `status: ${receipt.statusFile}`, ...(receipt.logFile ? [`log: ${receipt.logFile}`] : [])]);
  return ['failed', 'interrupted'].includes(receipt.state) ? 1 : 0;
}

async function cmdTrainCancel(opts, baseDir, rest) {
  const [jobId, requestId] = rest;
  if (!jobId || !requestId) throw new UsageError('usage: tmf train-cancel <jobId> <requestId>');
  const receipt = await cancelBackgroundTraining({ baseDir, configFile: opts.config ?? null, jobId, requestId });
  out(opts, receipt, [`training request ${requestId}: ${receipt.state}`]);
  return 0;
}

async function main(argv) {
  let opts;
  try {
    opts = parseArgs(argv);
  } catch (err) {
    console.error(err.message);
    console.error(USAGE);
    return 2;
  }
  // `-h/--help` and `-v/--version` short-circuit before dispatch so that
  // `tmf init --help` prints usage instead of actually writing files.
  if (opts.help) { console.log(USAGE); return 0; }
  if (opts.version) { console.log(VERSION); return 0; }
  const [command = 'help', ...rest] = opts._;
  const baseDir = path.resolve(opts.cwd ?? process.cwd());
  switch (command) {
    case 'help': console.log(USAGE); return 0;
    case 'version': console.log(VERSION); return 0;
    case 'doctor': return cmdDoctor(opts, baseDir);
    case 'init': return cmdInit(opts, baseDir);
    case 'config': return cmdConfig(opts, baseDir, rest);
    case 'backends': return cmdBackends(opts);
    case 'synth': return cmdSynth(opts, baseDir);
    case 'train': return cmdTrain(opts, baseDir);
    case 'evaluate': case 'eval': return cmdEvaluate(opts, baseDir);
    case 'train-status': return cmdBackgroundStatus(opts, baseDir, rest, 'train');
    case 'evaluate-status': return cmdBackgroundStatus(opts, baseDir, rest, 'evaluate');
    case 'train-cancel': return cmdTrainCancel(opts, baseDir, rest);
    case 'quality-contract': return cmdQualityContract(opts);
    case 'quality-prepare': return cmdQualityPrepare(opts, baseDir);
    case 'quality-finalize': return cmdQualityFinalize(opts, baseDir);
    case 'export': return cmdExport(opts, baseDir);
    case 'status': return cmdStatus(opts, baseDir, rest);
    default:
      console.error(`unknown command: ${command}`);
      console.error(USAGE);
      return 2;
  }
}

main(process.argv.slice(2)).then((code) => process.exit(code), (err) => {
  if (err instanceof ConfigError) {
    console.error(err.message);
    console.error(formatConfigErrors(err.errors));
    process.exit(1);
  }
  console.error(err?.stack ?? String(err));
  process.exit(1);
});
