// The OpenClaw host plugin surface. These assertions are what keep
// openclaw.plugin.json, the published tool list and the adapter action table
// from drifting apart; the host itself is not imported, so the suite runs in a
// clean checkout with no `openclaw` package installed.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { ACTION_SPECS, CONTRACT_VERSION } from '../src/integrations/openclaw.mjs';
import { PARAM_SCHEMA, TOOL_PREFIX, buildTools, parametersFor, toolNameFor } from '../integrations/openclaw-plugin/tools.mjs';

const manifest = JSON.parse(readFileSync(new URL('../openclaw.plugin.json', import.meta.url), 'utf8'));

test('the host manifest declares exactly the adapter actions, tmf_-prefixed', () => {
  const expected = Object.keys(ACTION_SPECS).map(toolNameFor);
  assert.deepEqual(manifest.contracts.tools, expected);
  assert.ok(expected.every((name) => name.startsWith(TOOL_PREFIX)));
});

test('package.json points the host at the plugin entry and ships the manifest', () => {
  const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
  assert.deepEqual(pkg.openclaw.extensions, ['./integrations/openclaw-plugin/index.mjs']);
  assert.ok(pkg.files.includes('openclaw.plugin.json'), 'manifest must be published');
  assert.ok(pkg.files.includes('integrations/openclaw-plugin/'), 'entry must be published');
});

test('the plugin faces are reachable as package subpaths once installed', () => {
  const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
  // A tarball install resolves these by package name; without the exports
  // entries Node answers ERR_PACKAGE_PATH_NOT_EXPORTED.
  assert.equal(pkg.exports['./integrations/openclaw-plugin'], './integrations/openclaw-plugin/index.mjs');
  assert.equal(pkg.exports['./integrations/openclaw-plugin/tools'], './integrations/openclaw-plugin/tools.mjs');
});

test('every adapter param has a JSON schema and no schema is dead weight', () => {
  const used = new Set(Object.values(ACTION_SPECS).flatMap((spec) => spec.params));
  for (const param of used) assert.ok(PARAM_SCHEMA[param], `missing schema for ${param}`);
  for (const param of Object.keys(PARAM_SCHEMA)) assert.ok(used.has(param), `unused schema for ${param}`);
});

test('generated tool parameters are closed and carry the adapter required list', () => {
  for (const [action, spec] of Object.entries(ACTION_SPECS)) {
    const schema = parametersFor(action);
    assert.equal(schema.additionalProperties, false);
    assert.deepEqual(Object.keys(schema.properties), spec.params);
    assert.deepEqual(schema.required ?? [], spec.required ?? []);
  }
});

test('tools forward to the adapter and honour a read-only mount', async () => {
  const tools = buildTools((t) => t, { config: { allowCompute: false } });
  assert.deepEqual(tools.map((t) => t.name), manifest.contracts.tools);

  const doctor = tools.find((t) => t.name === 'tmf_doctor');
  const ok = await doctor.execute({});
  assert.equal(ok.ok, true);
  assert.equal(ok.contractVersion, CONTRACT_VERSION);
  assert.equal(ok.action, 'doctor');
  assert.ok(Array.isArray(ok.result.checks) && ok.result.checks.length > 0);

  const train = tools.find((t) => t.name === 'tmf_train');
  const refused = await train.execute({ jobId: 'no-such-job' });
  assert.equal(refused.ok, false);
  assert.equal(refused.error.code, 'compute-not-authorized');
});

test('unknown params are refused through the tool surface, not the schema alone', async () => {
  const tools = buildTools((t) => t, { config: {} });
  const status = tools.find((t) => t.name === 'tmf_status');
  const bad = await status.execute({ jobId: 'x', nope: 1 });
  assert.equal(bad.ok, false);
  assert.equal(bad.error.code, 'unknown-params');
});
