// Optional process host for campaigns. It does not depend on an agent harness;
// systemd user units are preferred on Linux, with a detached-process fallback.
import path from 'node:path';
import { open, mkdir, stat, unlink, readFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { ensureDir, readJsonIfExists, writeJson } from '../core/fsx.mjs';
import { sha256Json } from '../core/hash.mjs';
import { readJobState } from '../jobs/runner.mjs';
import { launchBackgroundWorker } from '../jobs/async-evaluate.mjs';
import { loadCampaign, campaignStatus } from './campaign.mjs';

const workerFile = fileURLToPath(new URL('./campaign-worker.mjs', import.meta.url));
const execFileAsync = promisify(execFile);

async function active(launch) {
  if (!launch) return false;
  if (launch.launcher === 'systemd' && launch.unit) {
    try {
      const { stdout } = await execFileAsync('systemctl', ['--user', 'is-active', launch.unit], { timeout: 5000 });
      return stdout.trim() === 'active';
    } catch { return false; }
  }
  if (launch.pid) {
    try {
      process.kill(launch.pid, 0);
      if (process.platform === 'linux') {
        const cmdline = await readFile(`/proc/${launch.pid}/cmdline`, 'utf8');
        return cmdline.includes(workerFile);
      }
      return true;
    } catch { return false; }
  }
  return false;
}

function location(loaded) {
  const dir = path.join(loaded.runRoot, 'campaigns', loaded.plan.id);
  return { dir, launchFile: path.join(dir, 'controller-launch.json'), lockFile: path.join(dir, 'controller-start.lock') };
}

async function withStartLock(loc, fn) {
  await ensureDir(loc.dir);
  let lock;
  try { lock = await open(loc.lockFile, 'wx'); }
  catch (err) {
    if (err.code !== 'EEXIST') throw err;
    const age = Date.now() - (await stat(loc.lockFile)).mtimeMs;
    if (age > 30000) throw new Error('stale campaign start lock: inspect it before retrying');
    throw new Error('campaign start already in progress');
  }
  try { return await fn(); }
  finally { await lock.close(); await unlink(loc.lockFile); }
}

/** Deduplicate starts; an interrupted worker can be restarted explicitly. */
export async function startCampaign({ planFile, resumeCancelled = false } = {}) {
  const loaded = await loadCampaign(planFile);
  const loc = location(loaded);
  return withStartLock(loc, async () => {
    const inputDigest = sha256Json(loaded.input);
    const state = await readJobState(path.join(loaded.runRoot, 'campaigns'), loaded.plan.id);
    if (state && state.inputDigest !== inputDigest) throw new Error('campaign inputs changed; use a new campaign id');
    const previous = await readJsonIfExists(loc.launchFile);
    if (state && ['succeeded', 'failed', 'budget_exceeded'].includes(state.state)) {
      return { campaignId: loaded.plan.id, requestId: inputDigest.slice(0, 24), state: state.state,
        launch: previous, reused: true };
    }
    if (state?.state === 'cancelled' && !resumeCancelled) {
      return { campaignId: loaded.plan.id, requestId: inputDigest.slice(0, 24), state: 'cancelled',
        launch: previous, reused: true };
    }
    if (previous && (Date.now() - Date.parse(previous.startedAt) < 10000 || await active(previous))) {
      return { campaignId: loaded.plan.id, requestId: inputDigest.slice(0, 24), state: state?.state ?? 'starting',
        launch: previous, reused: true };
    }
    const attempt = (previous?.attempt ?? 0) + 1;
    const attemptDir = path.join(loc.dir, `controller-attempt-${attempt}`);
    await mkdir(attemptDir);
    const requestFile = path.join(attemptDir, 'request.json');
    const statusFile = path.join(attemptDir, 'worker-result.json');
    await writeJson(requestFile, { planFile: loaded.file, inputDigest, attempt, resumeCancelled });
    const requestId = inputDigest.slice(0, 24);
    const launch = await launchBackgroundWorker(workerFile, requestFile, statusFile,
      `tmf-campaign-${requestId}-${attempt}`, path.join(attemptDir, 'worker.log'));
    const record = { ...launch, attempt, requestId, campaignId: loaded.plan.id,
      startedAt: new Date().toISOString(), workerResultFile: statusFile,
      logFile: path.join(attemptDir, 'worker.log') };
    await writeJson(loc.launchFile, record);
    return { campaignId: loaded.plan.id, requestId, state: 'starting', launch: record, reused: false };
  });
}

export async function backgroundCampaignStatus({ planFile } = {}) {
  const loaded = await loadCampaign(planFile);
  const loc = location(loaded);
  const result = await campaignStatus({ planFile });
  const launch = await readJsonIfExists(loc.launchFile);
  const workerResult = launch ? await readJsonIfExists(launch.workerResultFile) : null;
  const workerActive = workerResult ? false : await active(launch);
  const state = result.state?.state ?? (launch ? 'starting' : 'not_started');
  const missingWorker = launch && !workerActive && !workerResult &&
    Date.now() - Date.parse(launch.startedAt) >= 10000;
  return { campaignId: loaded.plan.id,
    state: workerResult?.state === 'failed' ? 'failed'
      : ['running', 'starting'].includes(state) && missingWorker ? 'interrupted' : state,
    run: result.state, decision: result.state?.state === 'succeeded'
      ? (await readJsonIfExists(path.join(loc.dir, 'result.json')))?.result ?? null : null,
    launch, workerResult, workerActive };
}

export async function cancelCampaign({ planFile } = {}) {
  const status = await backgroundCampaignStatus({ planFile });
  if (!status.workerActive || !['running', 'starting'].includes(status.state)) return status;
  const { launch } = status;
  if (launch.launcher === 'systemd') {
    await execFileAsync('systemctl', ['--user', 'kill', '--signal=TERM', launch.unit], { timeout: 5000 });
  } else if (launch.pid) process.kill(launch.pid, 'SIGTERM');
  return { ...status, cancelRequested: true };
}
