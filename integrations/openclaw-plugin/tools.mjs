// Tool surface for the OpenClaw host plugin, kept free of any `openclaw`
// import so the package's own test suite can assert it without the host
// package installed. index.mjs adds nothing but the host registration.

import { ACTION_SPECS, createOpenClawAdapter } from '../../src/integrations/openclaw.mjs';

/** JSON Schema fragment per adapter param. Keyed by param name, not action. */
export const PARAM_SCHEMA = {
  configFile: { type: 'string', description: 'Foundry config file path.' },
  requireConfig: { type: 'boolean', description: 'Fail when no config file is present.' },
  jobId: { type: 'string', description: 'Durable foundry job id.' },
  auditTail: { type: 'number', description: 'How many audit-log lines to return.' },
  outDir: { type: 'string', description: 'Output directory.' },
  seed: { type: 'number', description: 'Deterministic seed.' },
  count: { type: 'number', description: 'Number of synthetic records to write.' },
  profile: { type: 'string', description: 'Synthetic corpus profile name.' },
  resume: { type: 'boolean', description: 'Resume an interrupted training job.' },
  resumeCancelled: { type: 'boolean', description: 'Resume a cancelled training job.' },
  auditEveryStep: { type: 'boolean', description: 'Append an audit entry per training step.' },
  gates: { type: 'string', description: 'Evaluation gate expression.' },
  reportFile: { type: 'string', description: 'Where to write the evaluation report.' },
  maxNewTokens: { type: 'number', description: 'Generation budget per eval sample.' },
  background: { type: 'boolean', description: 'Run training or evaluation in an independent worker and return a durable request id.' },
  retry: { type: 'boolean', description: 'Retry the same failed or interrupted background request; never duplicates a running request.' },
  requestId: { type: 'string', description: 'Background request id returned by tmf_train or tmf_evaluate.' },
  evalReport: { type: 'string', description: 'Evaluation report consumed by export.' },
  suiteFile: { type: 'string', description: 'Frozen 120-question Chinese screenplay suite JSON.' },
  scriptsDir: { type: 'string', description: 'Pinned screenplay scorer scripts directory (defaults to bundled scripts).' },
  baseOutputs: { type: 'string', description: 'Full-suite base model outputs JSONL.' },
  daptOutputs: { type: 'string', description: 'Full-suite frozen DAPT reference outputs JSONL.' },
  candidateOutputs: { type: 'string', description: 'Full-suite Foundry candidate outputs JSONL.' },
  cleanDaptReport: { type: 'string', description: 'Frozen DAPT clean-validation loss report.' },
  cleanDaptModel: { type: 'string', description: 'Model name inside DAPT clean-validation report.' },
  stageFile: { type: 'string', description: 'Prepared screenplay quality stage JSON.' },
  judgeVerdicts: { type: 'array', items: { type: 'string' }, description: 'At least two NAME=FILE blind verdict JSONL files.' },
  cleanCandidateReport: { type: 'string', description: 'Candidate clean-validation loss report.' },
  cleanCandidateModel: { type: 'string', description: 'Model name inside candidate clean-validation report.' },
  qualityReport: { type: 'string', description: 'Predeclared external quality evidence report consumed by export.' },
  version: { type: 'string', description: 'Export version label.' },
  evaluateIfMissing: { type: 'boolean', description: 'Run evaluation when no report exists.' },
};

export const TOOL_PREFIX = 'tmf_';

/** `tmf_train` -> `train`; exported so the host test can assert the mapping. */
export function toolNameFor(action) {
  return `${TOOL_PREFIX}${action}`;
}

export function parametersFor(action) {
  const spec = ACTION_SPECS[action];
  const properties = {};
  for (const param of spec.params) {
    const schema = PARAM_SCHEMA[param];
    if (!schema) throw new Error(`no JSON schema declared for param "${param}" of action "${action}"`);
    properties[param] = schema;
  }
  return {
    type: 'object',
    additionalProperties: false,
    properties,
    ...(spec.required?.length ? { required: [...spec.required] } : {}),
  };
}

/**
 * Build the tool list from the adapter's own action table so adding an action
 * in the adapter publishes it here too.
 */
export function buildTools(tool, { config = {} } = {}) {
  const adapter = createOpenClawAdapter({
    baseDir: config.baseDir ?? process.cwd(),
    configFile: config.configFile ?? null,
    allowCompute: config.allowCompute ?? true,
  });
  return Object.keys(ACTION_SPECS).map((action) =>
    tool({
      name: toolNameFor(action),
      description: ACTION_SPECS[action].summary,
      parameters: parametersFor(action),
      execute: async (params) => adapter.invoke(action, params ?? {}),
    }),
  );
}

