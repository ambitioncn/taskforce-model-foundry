// Tests for the installable product surface: config validation, the public
// API, and the `tmf` CLI as an actual child process (which is the only way to
// prove exit codes and that `bin/tmf.mjs` is wired to the package).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const run = promisify(execFile);
const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..');
const cli = path.join(repo, 'bin', 'tmf.mjs');

async function tmf(args, { cwd = repo } = {}) {
  try {
    const { stdout, stderr } = await run(process.execPath, [cli, ...args], { cwd, timeout: 240000 });
    return { code: 0, stdout, stderr };
  } catch (err) {
    return { code: err.code ?? 1, stdout: err.stdout ?? '', stderr: err.stderr ?? '' };
  }
}

async function tmpdir(label) {
  return fs.mkdtemp(path.join(os.tmpdir(), `tmf-${label}-`));
}

test('config validation reports every problem with its path', async () => {
  const { validateConfig, exampleConfig } = await import('../src/config.mjs');
  const good = validateConfig(exampleConfig());
  assert.equal(good.ok, true, JSON.stringify(good.errors));
  assert.equal(good.config.format, 'tmf-config-v1');
  assert.ok(path.isAbsolute(good.config.runRootAbsolute));

  const bad = validateConfig({
    format: 'nope', mode: 'invalid', model: { maxVocab: 0 },
    data: { datasets: [{ name: 'a' }] }, budget: { maxTokens: -1, bogus: 2 }, extra: 1,
  });
  assert.equal(bad.ok, false);
  assert.equal(bad.config, null);
  const at = bad.errors.map((e) => e.at);
  for (const expected of ['format', 'mode', 'model.maxVocab', 'data.datasets[0].path', 'budget.maxTokens', 'budget.bogus', 'extra']) {
    assert.ok(at.includes(expected), `missing error for ${expected}; got ${at.join(',')}`);
  }
});

test('hyper keys are validated instead of silently dropped', async () => {
  const { validateConfig, exampleConfig } = await import('../src/config.mjs');
  // A knob the trainer does not read must not look accepted: warmupSteps is
  // spelled warmupRatio, and a typo used to pass validation and do nothing.
  const bad = validateConfig({
    ...exampleConfig(),
    hyper: { steps: 1.5, lr: 0, batchSize: 2, warmupSteps: 2, lora: 3 },
  });
  assert.equal(bad.ok, false);
  const at = bad.errors.map((e) => e.at);
  for (const expected of ['hyper.steps', 'hyper.lr', 'hyper.warmupSteps', 'hyper.lora']) {
    assert.ok(at.includes(expected), `missing error for ${expected}; got ${at.join(',')}`);
  }

  const good = validateConfig({
    ...exampleConfig(),
    hyper: { steps: 600, lr: 0.05, batchSize: 4, warmupRatio: 0.1, minLrRatio: 0.1, lora: { rank: 4 } },
  });
  assert.equal(good.ok, true, JSON.stringify(good.errors));
  assert.equal(good.config.hyper.steps, 600);
});

test('the step budget comes from the config, not the backend default', async () => {
  const { resolveStepBudget, DEFAULT_BUDGET } = await import('../src/train/backends/builtin-js.mjs');
  assert.deepEqual(
    resolveStepBudget({ budget: {}, hyper: { steps: 600 } }),
    { maxSteps: 600, source: 'hyper.steps' },
  );
  // An explicit trainBudget still wins over hyper.steps.
  assert.deepEqual(
    resolveStepBudget({ budget: { maxSteps: 50 }, hyper: { steps: 600 } }),
    { maxSteps: 50, source: 'trainBudget.maxSteps' },
  );
  // Only an unconstrained config falls back to the built-in cap.
  assert.deepEqual(
    resolveStepBudget({ budget: {}, hyper: {} }),
    { maxSteps: DEFAULT_BUDGET.maxSteps, source: 'default' },
  );
  // maxSteps: null means "no step cap" and must survive the hyper fallback.
  assert.deepEqual(
    resolveStepBudget({ budget: { maxSteps: null }, hyper: { steps: 600 } }),
    { maxSteps: null, source: 'trainBudget.maxSteps' },
  );
  assert.throws(() => resolveStepBudget({ budget: {}, hyper: { steps: -1 } }), /hyper\.steps/);
});

test('config modes are only those a backend can actually run', async () => {
  const { validateConfig, exampleConfig } = await import('../src/config.mjs');
  const { loadDefaultBackends, listBackends } = await import('../src/train/backends/index.mjs');
  await loadDefaultBackends();
  const advertised = new Set(listBackends().flatMap((b) => b.capabilities?.modes ?? []));
  for (const mode of ['lora', 'pretrain', 'full-finetune']) {
    assert.equal(validateConfig({ ...exampleConfig(), mode }).ok, true, `mode ${mode} should validate`);
    assert.ok(advertised.has(mode), `no backend advertises mode ${mode}`);
  }
  assert.equal(validateConfig({ ...exampleConfig(), mode: 'full' }).ok, false);
});

test('doctor fails loudly when the configured dataset is missing', async () => {
  const { doctor } = await import('../src/api.mjs');
  const dir = await tmpdir('doctor');
  const { exampleConfig } = await import('../src/config.mjs');
  await fs.writeFile(path.join(dir, 'tmf.config.json'), JSON.stringify(exampleConfig()));
  const report = await doctor({ baseDir: dir });
  assert.equal(report.ok, false);
  assert.ok(report.failed.includes('dataset:tasks'), report.failed.join(','));
  const check = report.checks.find((c) => c.id === 'dataset:tasks');
  assert.match(check.detail, /missing dataset file/);
});

test('tmf help/version/unknown map to stable exit codes', async () => {
  assert.equal((await tmf(['version'])).code, 0);
  const help = await tmf(['help']);
  assert.equal(help.code, 0);
  assert.match(help.stdout, /Usage: tmf <command>/);
  assert.equal((await tmf(['no-such-command'])).code, 2, 'unknown command is a usage error');
  assert.equal((await tmf(['--bogus-flag'])).code, 2, 'unknown option is a usage error');
  assert.equal((await tmf(['config', 'validate', '-c'])).code, 2, 'missing option value is a usage error');
});

test('--help/-h short-circuits every subcommand without side effects', async () => {
  for (const flag of ['--help', '-h']) {
    const res = await tmf([flag]);
    assert.equal(res.code, 0, res.stderr);
    assert.match(res.stdout, /Usage: tmf <command>/);
  }
  for (const flag of ['--version', '-v']) {
    const res = await tmf([flag]);
    assert.equal(res.code, 0, res.stderr);
    assert.doesNotMatch(res.stdout, /Usage: tmf <command>/);
  }

  // The regression: `init --help` used to fall through to the real init and
  // write a config plus a synthetic corpus into the caller's directory.
  const dir = await tmpdir('help-init');
  const init = await tmf(['init', '--help', '-C', dir, '--count', '80']);
  assert.equal(init.code, 0, init.stderr);
  assert.match(init.stdout, /Usage: tmf <command>/);
  assert.deepEqual(await fs.readdir(dir), [], 'init --help must not write anything');

  // Same for the other file-touching subcommands, and `--help` must win even
  // when the rest of the argv is nonsense that would otherwise fail.
  for (const args of [['synth'], ['train'], ['evaluate'], ['export'], ['doctor'], ['config', 'validate'], ['status', 'job-x']]) {
    const res = await tmf([...args, '--help', '-C', dir]);
    assert.equal(res.code, 0, `${args.join(' ')} --help: ${res.stderr}`);
    assert.match(res.stdout, /Usage: tmf <command>/);
  }
  assert.deepEqual(await fs.readdir(dir), [], 'no subcommand may write files under --help');
});

test('tmf init then doctor then train is a working first run', async () => {
  const dir = await tmpdir('init');
  const init = await tmf(['init', '-C', dir, '--count', '80']);
  assert.equal(init.code, 0, init.stderr);
  const configFile = path.join(dir, 'tmf.config.json');
  const config = JSON.parse(await fs.readFile(configFile, 'utf8'));
  assert.ok(config.data.datasets.length > 0);
  for (const dataset of config.data.datasets) {
    await fs.access(path.join(dir, dataset.path));
  }

  // init must never clobber an existing config.
  assert.equal((await tmf(['init', '-C', dir])).code, 1);

  const doctorRun = await tmf(['doctor', '-C', dir, '--json']);
  assert.equal(doctorRun.code, 0, doctorRun.stdout + doctorRun.stderr);
  assert.equal(JSON.parse(doctorRun.stdout).ok, true);

  // A real (small) post-training run through the installed CLI.
  config.hyper = { steps: 12, lr: 0.05, batchSize: 2, warmupRatio: 0.2 };
  await fs.writeFile(configFile, JSON.stringify(config));
  const train = await tmf(['train', '-C', dir, '--job', 'cli-demo', '--json']);
  assert.equal(train.code, 0, train.stdout + train.stderr);
  const result = JSON.parse(train.stdout);
  assert.equal(result.job.state, 'succeeded');
  assert.deepEqual(result.job.completedSteps.map((s) => s.name ?? s), ['data', 'base', 'train', 'finalize']);
  assert.ok(result.artifacts.metrics.parameterL2Delta > 0, 'training must really move the parameters');
  // hyper.steps is the only step knob `tmf init` writes, so the run must stop
  // where the config says -- not at the backend's built-in default.
  assert.equal(result.artifacts.metrics.stepBudget, 12);
  assert.equal(result.artifacts.metrics.stepBudgetSource, 'hyper.steps');
  assert.equal(result.artifacts.metrics.steps, 12);
  assert.equal(result.artifacts.metrics.stopReason, 'max_steps');
  await fs.access(result.artifacts.adapterCheckpoint);

  const status = await tmf(['status', 'cli-demo', '-C', dir, '--json']);
  assert.equal(status.code, 0, status.stderr);
  const statusReport = JSON.parse(status.stdout);
  assert.equal(statusReport.found, true);
  assert.equal(statusReport.state.state, 'succeeded');
  assert.ok(statusReport.auditCount > 0);

  const missing = await tmf(['status', 'no-such-job', '-C', dir]);
  assert.equal(missing.code, 1);
});

test('package manifest points at files that exist', async () => {
  const pkg = JSON.parse(await fs.readFile(path.join(repo, 'package.json'), 'utf8'));
  const targets = [pkg.main, pkg.bin.tmf, ...Object.values(pkg.exports)];
  for (const target of targets) {
    if (!target) continue;
    const file = path.resolve(repo, target);
    const found = await fs.access(file).then(() => true, () => false);
    assert.equal(found, true, `package.json references missing file: ${target}`);
  }
});

// The README tells CI users to run `tmf doctor --json`, so the severity of a
// config problem is part of that promise: a bad config is only a *failure* when
// the caller named the file. Assert the behaviour and that the README still
// says so, so neither side can drift away from the other.
test('doctor config severity matches what the README promises CI', async () => {
  const dir = await tmpdir('doctor-severity');
  await fs.writeFile(path.join(dir, 'tmf.config.json'), '{ not json');

  const discovered = await tmf(['doctor', '--json'], { cwd: dir });
  assert.equal(discovered.code, 0, 'a config found by discovery is a warning, not a failure');
  const report = JSON.parse(discovered.stdout);
  assert.equal(report.ok, true);
  const check = report.checks.find((c) => c.id === 'config');
  assert.equal(check.level, 'optional');
  assert.equal(check.ok, false);
  assert.ok(report.configErrors.length, 'the problem is still reported, just not fatal');

  const named = await tmf(['doctor', '--config', 'tmf.config.json'], { cwd: dir });
  assert.equal(named.code, 1, '--config promotes the config check to required');
  assert.match(named.stdout, /tmf doctor — FAILED/);
  assert.equal((await tmf(['doctor', '--config', 'absent.json'], { cwd: dir })).code, 1);
  assert.equal((await tmf(['config', 'validate'], { cwd: dir })).code, 1, 'config validate is the strict route');

  const readme = await fs.readFile(path.join(repo, 'README.md'), 'utf8');
  assert.match(readme, /`--config <file>`/, 'README must name the flag that makes a bad config fatal');
  assert.match(
    readme.replace(/\n/g, ' '),
    /tmf doctor` grades an absent or unreadable config as a \*warning\* and still exits `0`/,
    'README must keep documenting that a discovered bad config is not fatal',
  );
});
