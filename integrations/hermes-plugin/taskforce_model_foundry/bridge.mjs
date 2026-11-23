#!/usr/bin/env node
// Node side of the Hermes Python-plugin bridge.
//
// Hermes discovers directory plugins written in Python, but the Foundry
// adapter is the Node module src/integrations/hermes.mjs. Rather than
// restate the action table in Python, this bridge exposes exactly two
// line-oriented commands over the existing adapter, so adding an action in
// src/operations.mjs still surfaces it on Hermes:
//
//   specs            -> prints {"ok":true,"specs":[...openai function specs]}
//   call <tool> <json-args>  -> prints the adapter envelope as JSON
//
// Every outcome is JSON on stdout with a boolean `ok`; the process never
// exits non-zero for a tool-level failure, because the Python side has to
// hand the model readable text either way.

const FOUNDRY = process.env.TMF_FOUNDRY_ROOT || new URL('../../../', import.meta.url).pathname;

function out(obj) {
  process.stdout.write(`${JSON.stringify(obj)}\n`);
}

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  let hermes;
  try {
    hermes = await import(new URL('src/integrations/hermes.mjs', `file://${FOUNDRY.replace(/\/?$/, '/')}`).href);
  } catch (err) {
    out({ ok: false, code: 'adapter-unavailable', error: String(err && err.message || err), foundryRoot: FOUNDRY });
    return;
  }
  if (cmd === 'specs') {
    out({ ok: true, host: hermes.HOST, contractVersion: hermes.CONTRACT_VERSION, specs: hermes.toolSpecs() });
    return;
  }
  if (cmd === 'call') {
    const [tool, argsJson] = rest;
    const adapter = hermes.createHermesAdapter({ baseDir: process.env.TMF_WORKSPACE || process.cwd() });
    const envelope = await adapter.callTool(tool, argsJson === undefined ? '{}' : argsJson);
    out({ ok: envelope.ok !== false, envelope });
    return;
  }
  out({ ok: false, code: 'unknown-bridge-command', error: `unknown command: ${String(cmd)}` });
}

main().catch((err) => out({ ok: false, code: 'bridge-crash', error: String(err && err.message || err) }));
