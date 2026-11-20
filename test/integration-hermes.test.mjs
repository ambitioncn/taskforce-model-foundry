// Contract tests for the Hermes adapter: the tool-call envelope a Hermes
// gateway may depend on, the JSON-string argument path, and the refusals the
// gateway needs to be able to distinguish.
import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import fs from 'node:fs/promises';
import path from 'node:path';

import { createHermesAdapter, manifest, toolSpecs, register, CONTRACT_VERSION, TOOL_NAMES, TOOL_PREFIX, HOST } from '../src/integrations/hermes.mjs';
import { ACTION_NAMES } from '../src/integrations/openclaw.mjs';
import { PRODUCT } from '../src/api.mjs';
import { VERSION } from '../src/version.mjs';

async function tmpDir() {
  return await fs.mkdtemp(path.join(os.tmpdir(), 'tmf-hermes-'));
}

test('tool specs cover every action with a closed schema', () => {
  const specs = toolSpecs();
  assert.deepEqual(specs.map((s) => s.function.name), ACTION_NAMES.map((a) => `${TOOL_PREFIX}${a}`));
  for (const s of specs) {
    assert.equal(s.type, 'function');
    assert.ok(s.function.description.length > 0);
    assert.equal(s.function.parameters.type, 'object');
    assert.equal(s.function.parameters.additionalProperties, false);
    for (const [, schema] of Object.entries(s.function.parameters.properties)) {
      assert.ok(schema.description, 'every param is documented for the model');
    }
    for (const r of s.function.parameters.required) {
      assert.ok(Object.prototype.hasOwnProperty.call(s.function.parameters.properties, r));
    }
  }
  assert.deepEqual(specs.find((s) => s.function.name === 'tmf_status').function.parameters.required, ['jobId']);
});

test('manifest describes the Hermes transport and does not drift from OpenClaw actions', () => {
  const m = manifest();
  assert.equal(m.host, HOST);
  assert.equal(m.product, PRODUCT);
  assert.equal(m.toolkitVersion, VERSION);
  assert.equal(m.contractVersion, CONTRACT_VERSION);
  assert.equal(m.transport, 'tool-call');
  assert.equal(m.toolPrefix, TOOL_PREFIX);
  assert.deepEqual(m.tools.map((t) => t.action), ACTION_NAMES);
  assert.deepEqual(TOOL_NAMES, ACTION_NAMES.map((a) => `${TOOL_PREFIX}${a}`));
});

test('doctor answers with a readable content string and a structured result', async () => {
  const baseDir = await tmpDir();
  const adapter = createHermesAdapter({ baseDir });
  const res = await adapter.callTool('tmf_doctor', {});
  assert.equal(res.ok, true);
  assert.equal(res.host, HOST);
  assert.equal(res.tool, 'tmf_doctor');
  assert.equal(res.action, 'doctor');
  assert.equal(res.contractVersion, CONTRACT_VERSION);
  assert.ok(Array.isArray(res.result.checks) && res.result.checks.length > 0);
  assert.equal(typeof res.content, 'string');
  assert.deepEqual(JSON.parse(res.content), JSON.parse(JSON.stringify(res.result)));
});

test('arguments arriving as a JSON string are parsed like an object', async () => {
  const baseDir = await tmpDir();
  const adapter = createHermesAdapter({ baseDir });
  const fromString = await adapter.callTool('tmf_doctor', '{"requireConfig": false}');
  const fromObject = await adapter.callTool('tmf_doctor', { requireConfig: false });
  assert.equal(fromString.ok, true);
  assert.equal(fromObject.ok, true);
  assert.deepEqual(fromString.result.ok, fromObject.result.ok);
  const empty = await adapter.callTool('tmf_doctor', '   ');
  assert.equal(empty.ok, true);
  const noArgs = await adapter.callTool('tmf_doctor');
  assert.equal(noArgs.ok, true);
});

test('malformed argument payloads are stable errors, not crashes', async () => {
  const adapter = createHermesAdapter({ baseDir: await tmpDir() });
  const bad = await adapter.callTool('tmf_doctor', '{not json');
  assert.equal(bad.ok, false);
  assert.equal(bad.error.code, 'invalid-json');
  const arr = await adapter.callTool('tmf_doctor', '[1,2]');
  assert.equal(arr.error.code, 'invalid-params');
  const unknownTool = await adapter.callTool('tmf_nope', {});
  assert.equal(unknownTool.error.code, 'unknown-tool');
  const unprefixed = await adapter.callTool('doctor', {});
  assert.equal(unprefixed.error.code, 'unknown-tool');
  const notAString = await adapter.callTool(null, {});
  assert.equal(notAString.error.code, 'unknown-tool');
  for (const res of [bad, arr, unknownTool, unprefixed, notAString]) {
    assert.equal(res.contractVersion, CONTRACT_VERSION);
    assert.ok(res.content.startsWith(`ERROR ${res.error.code}:`));
  }
});

test('unknown and missing params are refused before any work happens', async () => {
  const adapter = createHermesAdapter({ baseDir: await tmpDir() });
  const unknown = await adapter.callTool('tmf_doctor', { nope: 1 });
  assert.equal(unknown.error.code, 'unknown-params');
  assert.match(unknown.error.message, /nope/);
  const missing = await adapter.callTool('tmf_status', {});
  assert.equal(missing.error.code, 'missing-params');
  assert.match(missing.error.message, /jobId/);
});

test('a read-only mount refuses every compute tool but still answers doctor', async () => {
  const adapter = createHermesAdapter({ baseDir: await tmpDir(), allowCompute: false });
  assert.equal(adapter.allowCompute, false);
  for (const tool of ['tmf_train', 'tmf_evaluate', 'tmf_export', 'tmf_synthesize']) {
    const res = await adapter.callTool(tool, {});
    assert.equal(res.ok, false, `${tool} must not run`);
    assert.equal(res.error.code, 'compute-not-authorized');
  }
  const doctor = await adapter.callTool('tmf_doctor', {});
  assert.equal(doctor.ok, true);
});

test('failures inside an action become error envelopes with the action code', async () => {
  const adapter = createHermesAdapter({ baseDir: await tmpDir() });
  const res = await adapter.callTool('tmf_status', { jobId: 'does-not-exist' });
  assert.equal(res.ok, false);
  assert.equal(typeof res.error.code, 'string');
  assert.ok(res.error.message.length > 0);
  assert.equal(res.contractVersion, CONTRACT_VERSION);
});

test('register wires every tool spec into a gateway that takes one tool at a time', async () => {
  const baseDir = await tmpDir();
  const registered = [];
  const adapter = register({ registerTool: (entry) => registered.push(entry) }, { baseDir, allowCompute: false });
  assert.deepEqual(registered.map((r) => r.spec.function.name), TOOL_NAMES);
  const doctorEntry = registered.find((r) => r.spec.function.name === 'tmf_doctor');
  const res = await doctorEntry.handler('{}');
  assert.equal(res.ok, true);
  assert.equal(res.tool, 'tmf_doctor');
  const trainEntry = registered.find((r) => r.spec.function.name === 'tmf_train');
  assert.equal((await trainEntry.handler({})).error.code, 'compute-not-authorized');
  assert.equal(adapter.host, HOST);
});

test('register also supports a gateway that registers the whole tool set at once', async () => {
  const baseDir = await tmpDir();
  let batch = null;
  register({ registerTools: (payload) => { batch = payload; } }, { baseDir });
  assert.deepEqual(batch.tools.map((t) => t.function.name), TOOL_NAMES);
  const res = await batch.handler('tmf_doctor', {});
  assert.equal(res.ok, true);
  // A gateway-free registration must still hand back a usable adapter.
  const bare = register(null, { baseDir });
  assert.equal((await bare.callTool('tmf_doctor', {})).ok, true);
});
