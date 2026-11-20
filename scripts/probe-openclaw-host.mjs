// Real-host probe: mount the foundry adapter as an actual OpenClaw tool plugin
// using the INSTALLED openclaw plugin-sdk (not a local mock), then execute a
// read-only tool call through the SDK-defined entry.
import { createRequire } from 'node:module';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const require = createRequire(import.meta.url);
const OC = process.env.OPENCLAW_PACKAGE_ROOT || dirname(require.resolve('openclaw/package.json'));
const WT = fileURLToPath(new URL('..', import.meta.url));
const { defineToolPlugin, getToolPluginMetadata } = await import(`${OC}/dist/plugin-sdk/tool-plugin.js`);
const { Type } = await import(`${OC}/node_modules/typebox/build/index.mjs`).catch(() => import(`${OC}/node_modules/typebox/index.js`));
const oc = await import(`${WT}/src/integrations/openclaw.mjs`);

let defined = [];
const out = { ok: true, steps: [] };
const step = (name, detail) => out.steps.push({ name, detail });

const mf = oc.manifest();
step('manifest', { host: mf.host, contractVersion: mf.contractVersion, actions: mf.actions.length });

// Read-only mount: compute-spending actions must be refused by the adapter.
const adapter = oc.createOpenClawAdapter({ baseDir: '/tmp/tmf-oc-probe', allowCompute: false });

const entry = defineToolPlugin({
  manifest: { name: 'taskforce-model-foundry', version: mf.version, description: 'Taskforce Model Foundry' },
  tools: () => defined = mf.actions.map((a) => ({
    name: `tmf_${a.name}`,
    description: a.summary,
    parameters: Type.Object(Object.fromEntries(a.params.map((p) => [p, Type.Optional(Type.Unknown())]))),
    execute: (params) => adapter.invoke(a.name, params ?? {}),
  })),
});

const meta = getToolPluginMetadata(entry);
step('sdk_metadata', {
  toolCount: meta.tools.length,
  names: meta.tools.map((t) => t.name),
  everyToolHasJsonSchema: meta.tools.every((t) => t.parameters && t.parameters.type === 'object'),
  everyToolHasDescription: meta.tools.every((t) => typeof t.description === 'string' && t.description.length > 0),
});

const byName = new Map(defined.map((t) => [t.name, t]));
const doctor = byName.get('tmf_doctor');
const r1 = await doctor.execute({}, {}, { toolCallId: 'probe-1' });
step('readonly_call_tmf_doctor', { ok: r1.ok, contractVersion: r1.contractVersion, action: r1.action, checks: r1.result?.checks?.length ?? null });

const r2 = await byName.get('tmf_train').execute({}, {}, { toolCallId: 'probe-2' });
step('compute_gate_tmf_train', { ok: r2.ok, code: r2.error?.code, message: r2.error?.message });

const r3 = await byName.get('tmf_status').execute({}, {}, { toolCallId: 'probe-3' });
step('missing_required_tmf_status', { ok: r3.ok, code: r3.error?.code });

const r4 = await byName.get('tmf_doctor').execute({ bogusParam: 1 }, {}, { toolCallId: 'probe-4' });
step('unknown_param_tmf_doctor', { ok: r4.ok, code: r4.error?.code });

const assertions = [
  ['sdk accepted all 6 tools', meta.tools.length === 6],
  ['tool names are tmf_<action>', meta.tools.every((t, i) => t.name === `tmf_${mf.actions[i].name}`)],
  ['sdk metadata carries no execute (static manifest only)', meta.tools.every((t) => typeof t.execute === 'undefined')],
  ['sdk compiled typebox params to JSON Schema objects', meta.tools.every((t) => t.parameters?.type === 'object')],
  ['read-only doctor call succeeded through sdk entry', r1.ok === true && r1.contractVersion === 1 && (r1.result?.checks?.length ?? 0) > 0],
  ['compute action refused under allowCompute:false', r2.ok === false && r2.error?.code === 'compute-not-authorized'],
  ['missing required param is a stable error', r3.ok === false && r3.error?.code === 'missing-params'],
  ['unknown param is a stable error', r4.ok === false && r4.error?.code === 'unknown-params'],
  ['no call ever threw', true],
];
out.assertions = assertions.map(([name, pass]) => ({ name, pass }));
out.ok = assertions.every(([, p]) => p);
console.log(JSON.stringify(out, null, 1));
process.exit(out.ok ? 0 : 1);
