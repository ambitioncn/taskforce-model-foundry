// The integration doc is a host-facing contract, so drift in it is a real
// defect: a host that mounts the adapter from a stale table supplies params the
// adapter then rejects. These tests parse docs/integration.md and hold its
// tables to the live adapters, so adding an action without documenting it fails
// here rather than at a host.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { ACTION_NAMES, ACTION_SPECS, CONTRACT_VERSION } from '../src/integrations/openclaw.mjs';
import { TOOL_NAMES, TOOL_PREFIX } from '../src/integrations/hermes.mjs';

const DOC = readFileSync(fileURLToPath(new URL('../docs/integration.md', import.meta.url)), 'utf8');

/** Rows of the first markdown table whose header starts with `firstColumn`. */
function tableRows(firstColumn) {
  const rows = [];
  let inTable = false;
  for (const line of DOC.split('\n')) {
    const cells = line.trim().startsWith('|')
      ? line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim())
      : null;
    if (!cells) {
      if (inTable) break;
      continue;
    }
    if (!inTable) {
      if (cells[0] === firstColumn) inTable = true;
      continue;
    }
    if (cells.every((c) => /^-+$/.test(c))) continue;
    rows.push(cells);
  }
  assert.ok(rows.length, `no table found with first column "${firstColumn}"`);
  return rows;
}

const code = (cell) => cell.split(',').map((c) => c.trim().replace(/^`|`$/g, '')).filter((c) => c && c !== '—');

test('doc action table lists every action, tool, compute flag and param', () => {
  const rows = tableRows('OpenClaw action');
  assert.deepEqual(rows.map((r) => code(r[0])[0]), [...ACTION_NAMES]);

  for (const row of rows) {
    const [actionCell, toolCell, computeCell, requiredCell, paramsCell] = row;
    const action = code(actionCell)[0];
    const spec = ACTION_SPECS[action];

    assert.equal(code(toolCell)[0], `${TOOL_PREFIX}${action}`, `${action}: tool name`);
    assert.ok(TOOL_NAMES.includes(`${TOOL_PREFIX}${action}`), `${action}: tool registered`);
    assert.equal(computeCell, spec.compute ? 'yes' : 'no', `${action}: compute flag`);
    assert.deepEqual(code(requiredCell), [...(spec.required ?? [])], `${action}: required params`);
    assert.deepEqual(code(paramsCell), [...spec.params], `${action}: accepted params`);
  }
});

test('doc error-code table covers every code the adapters can emit', () => {
  const documented = new Set(tableRows('Code').map((r) => code(r[0])[0]));
  const emitted = new Set();
  for (const file of ['../src/integrations/openclaw.mjs', '../src/integrations/hermes.mjs']) {
    const src = readFileSync(fileURLToPath(new URL(file, import.meta.url)), 'utf8');
    for (const m of src.matchAll(/fail\([^,]+,\s*'([a-z-]+)'/g)) emitted.add(m[1]);
    for (const m of src.matchAll(/\?\s*'(config-error|action-failed)'/g)) emitted.add(m[1]);
  }
  assert.ok(emitted.size >= 8, `expected to find the adapter codes, found ${[...emitted].join(', ')}`);
  for (const c of emitted) assert.ok(documented.has(c), `error code "${c}" is not documented`);
});

test('doc states the live contract version and both subpath specifiers', () => {
  assert.match(DOC, new RegExp(`currently \`${CONTRACT_VERSION}\``));
  assert.match(DOC, /taskforce-model-foundry\/integrations\/openclaw/);
  assert.match(DOC, /taskforce-model-foundry\/integrations\/hermes/);
});

// The specifier table is the only place a host learns which entry points exist.
// An export added or renamed without touching the table leaves a host importing
// a path that throws ERR_PACKAGE_PATH_NOT_EXPORTED, so hold the table to the
// live `exports` map in both directions.
test('specifier table matches the package exports map exactly', () => {
  const pkg = JSON.parse(readFileSync(
    fileURLToPath(new URL('../package.json', import.meta.url)), 'utf8'));
  const documented = new Set(tableRows('Specifier').map((r) => code(r[0])[0]));
  const exported = new Set(Object.keys(pkg.exports)
    // package.json is a resolution detail for the plugin copy, documented in
    // prose in §3.1 rather than as a host entry point.
    .filter((s) => s !== './package.json')
    .map((s) => s === '.' ? pkg.name : `${pkg.name}/${s.replace(/^\.\//, '')}`));
  assert.deepEqual([...documented].sort(), [...exported].sort());
});

test('doc specifier table matches the package exports map', () => {
  const pkg = JSON.parse(
    readFileSync(fileURLToPath(new URL('../package.json', import.meta.url)), 'utf8'),
  );
  const documented = tableRows('Specifier').map((r) => code(r[0])[0]);
  const expected = Object.keys(pkg.exports)
    .filter((k) => k !== './package.json')
    .map((k) => (k === '.' ? 'taskforce-model-foundry' : `taskforce-model-foundry${k.slice(1)}`));
  assert.deepEqual(documented, expected, 'specifier table drifted from exports map');
  // ./package.json is load-bearing for the Hermes plugin copy, so the doc must
  // account for it instead of implying the table is the whole exports map.
  assert.match(DOC, /`package\.json` is a fifth, deliberate `exports` entry/);
  assert.doesNotMatch(DOC, /and four\s+subpath exports/);
});
