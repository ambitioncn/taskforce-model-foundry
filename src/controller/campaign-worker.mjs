import { readJson, writeJson } from '../core/fsx.mjs';
import { sha256Json } from '../core/hash.mjs';
import { loadCampaign, runCampaign } from './campaign.mjs';

const [, , requestFile, resultFile] = process.argv;
if (!requestFile || !resultFile) throw new Error('campaign worker requires request and result paths');
const request = await readJson(requestFile);
const controller = new AbortController();
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => controller.abort(new Error(`operator cancelled with ${signal}`)));
}
try {
  const loaded = await loadCampaign(request.planFile);
  if (sha256Json(loaded.input) !== request.inputDigest) throw new Error('campaign inputs changed after submission');
  const state = await runCampaign({ planFile: request.planFile, signal: controller.signal,
    resumeCancelled: request.resumeCancelled });
  await writeJson(resultFile, { state: state.state, attempt: request.attempt,
    resultDigest: sha256Json(state.result), finishedAt: new Date().toISOString() });
} catch (err) {
  await writeJson(resultFile, { state: controller.signal.aborted ? 'cancelled' : 'failed',
    attempt: request.attempt, error: { code: err?.code ?? 'CAMPAIGN_WORKER_FAILED', message: err?.message ?? String(err) },
    finishedAt: new Date().toISOString() });
  process.exitCode = controller.signal.aborted ? 130 : 1;
}
