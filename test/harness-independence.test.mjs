import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ACTION_NAMES, ACTION_SPECS, PARAM_SCHEMA, parametersFor, createOperationClient } from '../src/operations.mjs';
import { toolSpecs } from '../src/integrations/hermes.mjs';
import { buildTools } from '../integrations/openclaw-plugin/tools.mjs';

test('the operation contract depends on no agent harness and has typed params', () => {
  const src = readFileSync(new URL('../src/operations.mjs', import.meta.url), 'utf8');
  assert.doesNotMatch(src, /from\s+['"](?:openclaw|hermes|\.\/integrations\/(?:openclaw|hermes))/);
  const used = new Set(Object.values(ACTION_SPECS).flatMap((spec) => spec.params));
  assert.deepEqual([...Object.keys(PARAM_SCHEMA)].sort(), [...used].sort());
  for (const action of ACTION_NAMES) {
    for (const schema of Object.values(parametersFor(action).properties)) {
      assert.ok(schema.type, `${action}: every param needs a JSON Schema type`);
    }
  }
});

test('optional harness adapters expose the same typed operation schema', () => {
  const openclaw = buildTools((entry) => entry);
  const hermes = toolSpecs();
  assert.deepEqual(openclaw.map((entry) => entry.name), hermes.map((entry) => entry.function.name));
  for (const [index, action] of ACTION_NAMES.entries()) {
    assert.deepEqual(openclaw[index].parameters.properties, parametersFor(action).properties);
    assert.deepEqual(hermes[index].function.parameters.properties, parametersFor(action).properties);
  }
});

test('the neutral client works without a harness adapter', async () => {
  const client = createOperationClient({ allowCompute: false });
  assert.equal((await client.invoke('train', {})).error.code, 'compute-not-authorized');
  assert.equal((await client.invoke('campaign_start', { planFile: 'campaign.json' })).error.code, 'compute-not-authorized');
  assert.equal((await client.invoke('campaign_cancel', { planFile: 'campaign.json' })).error.code, 'compute-not-authorized');
  assert.equal((await client.invoke('not-an-action', {})).error.code, 'unknown-action');
  assert.equal((await client.invoke('__proto__', {})).error.code, 'unknown-action');
});
