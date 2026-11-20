// Contract tests for the OpenClaw adapter: the envelope an OpenClaw host is
// allowed to depend on, and the refusals it needs to be able to distinguish.
import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import fs from 'node:fs/promises';
import path from 'node:path';

import { createOpenClawAdapter, manifest, register, CONTRACT_VERSION, ACTION_NAMES } from '../src/integrations/openclaw.mjs';
import { PRODUCT } from '../src/api.mjs';
import { VERSION } from '../src/version.mjs';

async function tmpDir() {
  return await fs.mkdtemp(path.join(os.tmpdir(), 'tmf-openclaw-'));
}

test('manifest declares the product, contract version and every action', () => {
  const m = manifest();
  assert.equal(m.host, 'openclaw');
  assert.equal(m.product, PRODUCT);
  assert.equal(m.toolkitVersion, VERSION);
  assert.equal(m.contractVersion, CONTRACT_VERSION);
  assert.equal(m.transport, 'in-process');
  assert.deepEqual(m.actions.map((a) => a.name), ACTION_NAMES);
  for (const a of m.actions) {
    assert.ok(a.summary.length > 10, `${a.name} needs a summary`);
    assert.ok(Array.isArray(a.params));
    assert.equal(typeof a.compute, 'boolean');
  }
  // The manifest must be serializable as-is: a host ships it over JSON.
  assert.deepEqual(JSON.parse(JSON.stringify(m)), m);
});

test('compute-spending actions are flagged and read-only actions are not', () => {
  const byName = Object.fromEntries(manifest().actions.map((a) => [a.name, a]));
  assert.equal(byName.doctor.compute, false);
  assert.equal(byName.status.compute, false);
  for (const name of ['train', 'evaluate', 'export', 'synthesize']) {
    assert.equal(byName[name].compute, true, `${name} must be marked as compute`);
  }
});

test('doctor runs through the adapter and returns the api record', async () => {
  const dir = await tmpDir();
  const adapter = createOpenClawAdapter({ baseDir: dir });
  const res = await adapter.invoke('doctor');
  assert.equal(res.ok, true);
  assert.equal(res.action, 'doctor');
  assert.equal(res.contractVersion, CONTRACT_VERSION);
  assert.equal(res.result.product, PRODUCT);
  assert.ok(Array.isArray(res.result.checks) && res.result.checks.length > 0);
  assert.ok(res.result.checks.some((c) => c.id === 'node-version'));
});

test('unknown action and malformed input are refused with stable codes, not throws', async () => {
  const adapter = createOpenClawAdapter({ baseDir: await tmpDir() });
  const unknown = await adapter.invoke('fine-tune-everything');
  assert.equal(unknown.ok, false);
  assert.equal(unknown.error.code, 'unknown-action');
  assert.match(unknown.error.message, /known actions/);

  assert.equal((await adapter.invoke('')).error.code, 'invalid-action');
  assert.equal((await adapter.invoke('doctor', [])).error.code, 'invalid-params');
  assert.equal((await adapter.invoke('doctor', null)).error.code, 'invalid-params');
});

test('unknown params are rejected rather than silently dropped', async () => {
  const adapter = createOpenClawAdapter({ baseDir: await tmpDir() });
  const res = await adapter.invoke('doctor', { requireConfig: false, learningRate: 0.5 });
  assert.equal(res.ok, false);
  assert.equal(res.error.code, 'unknown-params');
  assert.match(res.error.message, /learningRate/);
});

test('required params are enforced before any work starts', async () => {
  const adapter = createOpenClawAdapter({ baseDir: await tmpDir() });
  const res = await adapter.invoke('status', {});
  assert.equal(res.ok, false);
  assert.equal(res.error.code, 'missing-params');
  assert.match(res.error.message, /jobId/);
});

test('a read-only mount refuses compute but still answers doctor', async () => {
  const dir = await tmpDir();
  const adapter = createOpenClawAdapter({ baseDir: dir, allowCompute: false });
  for (const name of ['train', 'evaluate', 'export', 'synthesize']) {
    const res = await adapter.invoke(name, {});
    assert.equal(res.ok, false, `${name} must be refused`);
    assert.equal(res.error.code, 'compute-not-authorized');
  }
  assert.equal((await adapter.invoke('doctor')).ok, true);
});

test('failures inside the api surface as envelopes with a code', async () => {
  const dir = await tmpDir();
  const adapter = createOpenClawAdapter({ baseDir: dir });
  // No config file anywhere: status has to resolve one and cannot.
  const res = await adapter.invoke('status', { jobId: 'nope', configFile: path.join(dir, 'missing.json') });
  assert.equal(res.ok, false);
  assert.equal(typeof res.error.code, 'string');
  assert.ok(res.error.code.length > 0);
  assert.ok(res.error.message.length > 0);
  assert.deepEqual(JSON.parse(JSON.stringify(res)), res);
});

test('register hands an OpenClaw host a provider bound to the same invoke', async () => {
  const dir = await tmpDir();
  const registered = [];
  const host = { registerToolProvider: (p) => registered.push(p) };
  const adapter = register(host, { baseDir: dir });
  assert.equal(registered.length, 1);
  assert.equal(registered[0].name, PRODUCT);
  assert.deepEqual(registered[0].manifest.actions.map((a) => a.name), ACTION_NAMES);
  const res = await registered[0].invoke('doctor');
  assert.equal(res.ok, true);
  assert.equal(adapter.host, 'openclaw');
  // A host without the hook still gets a usable adapter back.
  assert.equal((await register(null, { baseDir: dir }).invoke('doctor')).ok, true);
});
