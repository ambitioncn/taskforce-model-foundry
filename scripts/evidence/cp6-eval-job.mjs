// R4 acceptance run: the evaluation job scored against the real cp5
// checkpoints, on both gate paths (DEFAULT_GATES pass, impossible-threshold
// fail) so the refusal path has real output too.
import { writeSyntheticCorpus } from '../../src/data/synth.mjs';
import { runDataPipeline } from '../../src/data/pipeline.mjs';
import { buildTokenizer, encodeRecords } from '../../src/train/dataset.mjs';
import builtin from '../../src/train/backends/builtin-js.mjs';
import { DEFAULT_GATES } from '../../src/eval/gates.mjs';
import { runEvaluationJob, assertDeliverable } from '../../src/jobs/evaluate.mjs';
import { writeJson } from '../../src/core/fsx.mjs';

const outFile = process.env.TMF_EVIDENCE_OUT || '.tmf-run/cp6-eval-job-report.json';
const base = process.env.TMF_BASE || '.tmf-run/ckpt5/base.json';
const adapter = process.env.TMF_ADAPTER || '.tmf-run/ckpt5/adapter.json';

const corpus = await writeSyntheticCorpus('.tmf-run/corpus');
const pre = await runDataPipeline({
  datasets: corpus.pretrainDatasets,
  config: { split: { train: 0.9, validation: 0.1, holdout: 0, requireNonEmptyHoldout: false } },
});
const task = await runDataPipeline({ datasets: corpus.datasets });
const tokenizer = buildTokenizer([...pre.accepted, ...task.accepted], { maxVocab: 128 });
const encoded = {
  holdout: encodeRecords(task.splits.holdout, tokenizer, { maxSeqLen: 128 }).examples,
  validation: encodeRecords(task.splits.validation, tokenizer, { maxSeqLen: 128 }).examples,
};
const records = { holdout: task.splits.holdout, validation: task.splits.validation, train: task.splits.train };

const passing = await runEvaluationJob({
  backend: builtin,
  baseCheckpoint: base,
  adapterCheckpoint: adapter,
  encoded,
  records,
  gates: DEFAULT_GATES,
  lineage: { dataReport: task.report ?? null, corpus: 'synthetic (self-generated)', trainedBy: 'scripts/evidence/cp5-train-gates.mjs' },
  jobId: 'cp6-eval-job-default-gates',
});
let deliverable = null;
try { assertDeliverable(passing); deliverable = true; } catch { deliverable = false; }

const failing = await runEvaluationJob({
  backend: builtin,
  baseCheckpoint: base,
  adapterCheckpoint: adapter,
  encoded,
  records,
  gates: [
    { id: 'impossible_exact_match', metric: 'tuned.holdout.exactMatch.accuracy', comparator: 'gte', threshold: 1, description: 'deliberately unreachable: proves the refusal path' },
    ...DEFAULT_GATES,
  ],
  jobId: 'cp6-eval-job-impossible-gate',
});
let refused = false;
try { assertDeliverable(failing); } catch (err) { refused = err.code === 'EVAL_NOT_DELIVERABLE'; }

await writeJson(outFile, {
  generatedAt: new Date().toISOString(),
  reproduce: 'TMF_EVIDENCE_OUT=<out> node scripts/evidence/cp6-eval-job.mjs',
  passPath: { report: passing, assertDeliverable: deliverable },
  failPath: { verdict: failing.verdict, gates: failing.gates, assertDeliverableRefused: refused },
});

console.log('pass path:', passing.verdict.status, passing.gates.summary, 'deliverable=', deliverable);
console.log('  baseline holdout loss', passing.metrics.baseline.holdout.loss, 'exact', passing.metrics.baseline.holdout.exactMatch.accuracy);
console.log('  tuned    holdout loss', passing.metrics.tuned.holdout.loss, 'exact', passing.metrics.tuned.holdout.exactMatch.accuracy);
console.log('  improvement', JSON.stringify(passing.metrics.improvement));
console.log('  independence', JSON.stringify(passing.independence));
console.log('fail path:', failing.verdict.status, failing.gates.summary, 'refused=', refused);
console.log('report:', outFile);

if (!(passing.verdict.status === 'passed' && deliverable === true && failing.verdict.status === 'failed' && refused)) {
  console.error('R4 evidence incomplete');
  process.exit(3);
}
