import test from 'node:test';
import assert from 'node:assert/strict';
import { validateConfig } from '../src/config.mjs';
import { DEFAULT_GATES } from '../src/eval/gates.mjs';

const base = {
  format: 'tmf-config-v1',
  data: { datasets: [{ name: 't', path: 'x.jsonl', format: 'jsonl', license: 'internal', source: 'synthetic' }] },
};

test('eval.gates is absent by default so the built-in gates still apply', () => {
  const { ok, config } = validateConfig({ ...base });
  assert.equal(ok, true);
  assert.deepEqual(config.eval, {});
  assert.equal(config.eval.gates ?? DEFAULT_GATES, DEFAULT_GATES);
});

test('a declared gate list is normalized and reaches the config', () => {
  const { ok, config, errors } = validateConfig({
    ...base,
    eval: { gates: [{ id: 'loss', metric: 'tuned.holdout.loss', comparator: 'lte', threshold: 4, description: 'ceiling' }] },
  });
  assert.deepEqual(errors, []);
  assert.equal(ok, true);
  assert.equal(config.eval.gates.length, 1);
  assert.equal(config.eval.gates[0].threshold, 4);
});

test('an unknown comparator, a non-numeric threshold and a bare minSamples are all reported by path', () => {
  const { ok, errors } = validateConfig({
    ...base,
    eval: { gates: [{ id: 'g', metric: 'm', comparator: 'approx', threshold: 'low', minSamples: 4 }] },
  });
  assert.equal(ok, false);
  const at = errors.map((e) => e.at);
  assert.ok(at.includes('eval.gates[0].comparator'));
  assert.ok(at.includes('eval.gates[0].threshold'));
  assert.ok(at.includes('eval.gates[0].sampleMetric'));
});

test('an empty gate list is rejected rather than silently disabling every gate', () => {
  const { ok, errors } = validateConfig({ ...base, eval: { gates: [] } });
  assert.equal(ok, false);
  assert.ok(errors.some((e) => e.at === 'eval.gates' && /at least one gate/.test(e.message)));
});

test('duplicate gate ids and unknown eval keys are rejected', () => {
  const dup = validateConfig({
    ...base,
    eval: { gates: [{ id: 'g', metric: 'a', comparator: 'gte', threshold: 1 }, { id: 'g', metric: 'b', comparator: 'gte', threshold: 1 }] },
  });
  assert.equal(dup.ok, false);
  assert.ok(dup.errors.some((e) => /duplicate gate id/.test(e.message)));
  const unknown = validateConfig({ ...base, eval: { gates: [{ id: 'g', metric: 'a', comparator: 'gte', threshold: 1 }], strict: true } });
  assert.equal(unknown.ok, false);
  assert.ok(unknown.errors.some((e) => e.at === 'eval.strict'));
});
