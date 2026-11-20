// The Hermes directory plugin is Python, so its own regression suite lives in
// integrations/hermes-plugin/tests. This test makes `npm test` run that suite
// too, so the host-compatibility fixes (positional dispatch args, optional
// ctx.log) cannot regress unnoticed from the Node side of the repo.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const testsDir = path.join(repoRoot, 'integrations', 'hermes-plugin', 'tests');

function python() {
  for (const candidate of ['python3', 'python']) {
    const probe = spawnSync(candidate, ['--version'], { encoding: 'utf8' });
    if (probe.status === 0) return candidate;
  }
  return null;
}

test('hermes plugin python regression suite passes', (t) => {
  const bin = python();
  if (!bin) {
    t.skip('no python interpreter on PATH');
    return;
  }
  const run = spawnSync(bin, ['-m', 'unittest', 'discover', '-s', testsDir, '-p', 'test_*.py'], {
    cwd: repoRoot,
    encoding: 'utf8',
  });
  assert.equal(run.status, 0, `python plugin tests failed:\n${run.stdout}\n${run.stderr}`);
  // unittest reports to stderr; assert it actually collected tests rather than
  // silently discovering none.
  const report = `${run.stdout}${run.stderr}`;
  const ran = /Ran (\d+) tests?/.exec(report);
  assert.ok(ran, `could not find test count in output:\n${report}`);
  assert.ok(Number(ran[1]) >= 8, `expected at least 8 plugin tests, ran ${ran[1]}`);
});
