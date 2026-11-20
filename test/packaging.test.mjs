import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));

function filesUnder(abs) {
  const out = [];
  const walk = (p, rel) => {
    for (const name of readdirSync(p)) {
      const child = join(p, name);
      const childRel = rel ? `${rel}/${name}` : name;
      if (statSync(child).isDirectory()) walk(child, childRel);
      else out.push(childRel);
    }
  };
  walk(abs, '');
  return out;
}

// Entries in package.json "files" are the publish manifest. A declared entry that
// resolves to nothing (missing path, or a directory holding zero files) ships
// nothing and is dead weight that silently rots.
const included = pkg.files.filter((e) => !e.startsWith('!'));

test('every package.json files entry contributes at least one real file', () => {
  assert.ok(Array.isArray(pkg.files) && pkg.files.length > 0, 'files must be a non-empty array');
  const empty = [];
  for (const entry of included) {
    const abs = join(root, entry);
    assert.ok(existsSync(abs), `files entry does not exist on disk: ${entry}`);
    const count = statSync(abs).isDirectory() ? filesUnder(abs).length : 1;
    if (count === 0) empty.push(entry);
  }
  assert.deepEqual(empty, [], `files entries resolve to zero files: ${empty.join(', ')}`);
});

// main / bin / exports are the package's public entry points. Each must exist and
// be inside the publish manifest, otherwise the published tarball installs broken.
test('main, bin and exports targets exist and are covered by files', () => {
  const targets = new Set();
  if (pkg.main) targets.add(pkg.main);
  for (const v of Object.values(pkg.bin ?? {})) targets.add(v);
  for (const v of Object.values(pkg.exports ?? {})) targets.add(v);
  assert.ok(targets.size >= 5, `expected main+bin+exports targets, got ${targets.size}`);

  const covered = (rel) => included.some((entry) => {
    const e = entry.replace(/^\.\//, '');
    return e.endsWith('/') ? rel.startsWith(e) : rel === e;
  });

  for (const target of targets) {
    const rel = target.replace(/^\.\//, '');
    assert.ok(existsSync(join(root, rel)), `entry point missing on disk: ${target}`);
    assert.ok(covered(rel), `entry point not covered by package.json files: ${target}`);
  }
});

// Scratch/experiment scripts at the repo root are not part of the product surface.
test('no stray scratch scripts at repository root', () => {
  const stray = readdirSync(root)
    .filter((n) => /^(exp|scratch|tmp)\d*\.(mjs|js|cjs)$/.test(n));
  assert.deepEqual(stray, [], `remove scratch scripts from repo root: ${stray.join(', ')}`);
});

// docs/integration.md documents the Hermes directory plugin as a `cp -r` out of an
// installed checkout. A path the docs tell users to copy must actually ship in the
// tarball, otherwise the documented install is impossible from npm.
test('paths the integration doc tells users to copy are inside the publish manifest', () => {
  const doc = readFileSync(join(root, 'docs/integration.md'), 'utf8');
  // The doc writes the source as "$TMF_PKG"/<path> so one snippet covers both an npm
  // install and a checkout; strip that install-root variable to get the in-package path.
  const copied = [...doc.matchAll(/^cp -r\s+(\S+)/gm)]
    .map((m) => m[1].replace(/^"\$TMF_PKG"\/?/, '').replace(/^\$TMF_PKG\/?/, ''));
  assert.ok(copied.length > 0, 'expected at least one documented cp -r source path');
  for (const rel of copied) {
    assert.ok(existsSync(join(root, rel)), `documented path missing on disk: ${rel}`);
    const inManifest = included.some((entry) => {
      const e = entry.replace(/^\.\//, '').replace(/\/$/, '');
      return rel === e || rel.startsWith(`${e}/`);
    });
    assert.ok(inManifest, `documented path not covered by package.json files: ${rel}`);
  }
});

// The real tarball is the only authority on what an install gets: assert the plugin
// files are in it and that Python bytecode caches are not.
test('npm pack ships the hermes plugin and no python bytecode', () => {
  const res = spawnSync('npm', ['pack', '--dry-run', '--json'], { cwd: root, encoding: 'utf8' });
  assert.equal(res.status, 0, `npm pack failed: ${res.stderr}`);
  const entries = JSON.parse(res.stdout)[0].files.map((f) => f.path);
  for (const want of [
    'integrations/hermes-plugin/taskforce_model_foundry/plugin.yaml',
    'integrations/hermes-plugin/taskforce_model_foundry/__init__.py',
    'integrations/hermes-plugin/taskforce_model_foundry/bridge.mjs',
  ]) {
    assert.ok(entries.includes(want), `tarball missing ${want}`);
  }
  const junk = entries.filter((p) => p.endsWith('.pyc') || p.includes('__pycache__'));
  assert.deepEqual(junk, [], `tarball ships python bytecode: ${junk.join(', ')}`);
});

// A closed `exports` map makes `require.resolve('<pkg>/package.json')` throw, which is
// exactly the call docs/integration.md tells a host to use to locate the installed
// plugin source. Keep the subpath and the documented command in lockstep.
test('the installed package exposes the manifest the integration doc resolves', () => {
  assert.equal(pkg.exports['./package.json'], './package.json', 'exports must expose ./package.json');
  const doc = readFileSync(join(root, 'docs/integration.md'), 'utf8');
  assert.match(
    doc,
    /require\.resolve\('taskforce-model-foundry\/package\.json'\)/,
    'integration doc must document resolving the package root for an npm install',
  );
  const res = spawnSync(
    process.execPath,
    ['-e', "process.stdout.write(require('module').createRequire(process.argv[1] + '/').resolve('taskforce-model-foundry/package.json'))", join(root, 'x')],
    { encoding: 'utf8' },
  );
  assert.equal(res.status, 0, `documented require.resolve is not supported by exports: ${res.stderr}`);
  assert.equal(resolve(res.stdout), join(root, 'package.json'));
});

// A command the docs offer an installed user must be runnable there. `npm run`
// reads the *host's* package.json, so a bare `npm run demo` only works in a
// checkout; the installed copy needs the shipped script's own path.
const INSTALL_FACING_DOCS = ['README.md', 'docs/tutorial.md'];

for (const doc of INSTALL_FACING_DOCS) {
  test(`every ${doc} command an installed user is offered resolves inside an install`, () => {
    const text = readFileSync(join(root, doc), 'utf8');
    const scripts = pkg.scripts ?? {};

    const installPaths = [...text.matchAll(/node_modules\/taskforce-model-foundry\/([\w./-]+)/g)]
      .map((m) => m[1]);
    assert.ok(
      installPaths.includes('scripts/e2e-demo.sh'),
      `${doc} must show the demo script path that works from an install`,
    );
    for (const rel of installPaths) {
      assert.ok(existsSync(join(root, rel)), `${doc} install path missing on disk: ${rel}`);
      const inManifest = included.some((entry) =>
        entry.endsWith('/') ? rel.startsWith(entry) : entry === rel);
      assert.ok(inManifest, `${doc} install path not covered by package.json files: ${rel}`);
    }

    // Any `npm run <script>` the doc shows must be a real script here and must
    // be labelled as the checkout route, so an installed reader is not sent to it.
    for (const [, name] of text.matchAll(/npm run ([\w:-]+)/g)) {
      assert.ok(scripts[name], `${doc} documents a missing npm script: ${name}`);
      if (name === 'test') continue;
      const block = text.slice(0, text.indexOf(`npm run ${name}`));
      assert.match(
        block.slice(-200),
        /source checkout/i,
        `${doc} must mark \`npm run ${name}\` as the source-checkout route`,
      );
    }
  });
}