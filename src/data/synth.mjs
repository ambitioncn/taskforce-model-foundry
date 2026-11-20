// Synthetic corpus generator.
//
// Two corpora are produced from one seed:
//
//   * a pretraining corpus of generic enterprise prose, used to fit the base
//     model, and
//   * a task corpus in an instruction format the base model has never seen,
//     used for adapter post-training.
//
// Keeping the task format out of the pretraining distribution is deliberate:
// the gap between the two is what makes the post-training improvement real and
// measurable on a holdout set instead of a restatement of what the base already
// knew. Optional noise injection (duplicates, PII, junk records) exercises the
// data pipeline on inputs that actually need cleaning.

import path from 'node:path';
import { Rng } from '../core/rng.mjs';
import { writeJsonl, ensureDir } from '../core/fsx.mjs';

export const ROUTING_RULES = [
  { trigger: 'billing', queue: 'finance' },
  { trigger: 'outage', queue: 'platform' },
  { trigger: 'access', queue: 'security' },
  { trigger: 'hiring', queue: 'people' },
  { trigger: 'legal', queue: 'compliance' },
  { trigger: 'shipping', queue: 'logistics' },
];

export const PRIORITY_RULES = [
  { qualifier: 'critical', priority: 'p1' },
  { qualifier: 'urgent', priority: 'p1' },
  { qualifier: 'soon', priority: 'p2' },
  { qualifier: 'routine', priority: 'p3' },
  { qualifier: 'later', priority: 'p3' },
];

const NOUNS = ['refund', 'invoice', 'request', 'ticket', 'incident', 'review', 'renewal', 'audit', 'handoff', 'escalation'];
const TEAMS = ['finance', 'platform', 'security', 'people', 'compliance', 'logistics'];
const VERBS = ['reviewed', 'approved', 'escalated', 'closed', 'reopened', 'documented'];
const PERIODS = ['this week', 'last quarter', 'on monday', 'after standup', 'before the audit'];

export const TASK_PROMPT_HEADER = 'route\nsubject: ';
export const TASK_COMPLETION_PREFIX = '\nqueue=';

export function renderTaskPrompt(subject) {
  return `${TASK_PROMPT_HEADER}${subject}`;
}

export function renderTaskCompletion(queue, priority) {
  return `${TASK_COMPLETION_PREFIX}${queue}\npriority=${priority}\n`;
}

export function expectedRouting(subject) {
  const words = subject.trim().split(/\s+/);
  const rule = ROUTING_RULES.find((r) => r.trigger === words[0]);
  const priorityRule = PRIORITY_RULES.find((p) => words.includes(p.qualifier));
  return {
    queue: rule ? rule.queue : null,
    priority: priorityRule ? priorityRule.priority : 'p3',
  };
}

export function generateTaskRecords(count, seed = 'tmf-task') {
  const rng = new Rng(seed);
  const seen = new Set();
  const out = [];
  let guard = 0;
  while (out.length < count && guard < count * 50) {
    guard += 1;
    const rule = rng.pick(ROUTING_RULES);
    const priorityRule = rng.pick(PRIORITY_RULES);
    const noun = rng.pick(NOUNS);
    const extra = rng.pick(NOUNS);
    const subject = `${rule.trigger} ${noun} ${priorityRule.qualifier} ${extra}`;
    if (seen.has(subject)) continue;
    seen.add(subject);
    out.push({
      prompt: renderTaskPrompt(subject),
      completion: renderTaskCompletion(rule.queue, priorityRule.priority),
      meta: { subject, queue: rule.queue, priority: priorityRule.priority },
    });
  }
  return out;
}

export function generatePretrainDocs(count, seed = 'tmf-pretrain') {
  const rng = new Rng(seed);
  const out = [];
  for (let i = 0; i < count; i += 1) {
    const sentences = [];
    const n = 2 + rng.int(2);
    for (let s = 0; s < n; s += 1) {
      const team = rng.pick(TEAMS);
      const noun = rng.pick(NOUNS);
      const verb = rng.pick(VERBS);
      const trigger = rng.pick(ROUTING_RULES).trigger;
      const period = rng.pick(PERIODS);
      sentences.push(`the ${team} team ${verb} the ${trigger} ${noun} ${period}.`);
    }
    out.push({ text: sentences.join(' ') });
  }
  return out;
}

// Records that must be caught by the pipeline rather than trained on.
export function generateNoisyRecords(taskRecords, seed = 'tmf-noise') {
  const rng = new Rng(seed);
  const out = [];
  const pick = (i) => taskRecords[i % taskRecords.length];

  // Exact duplicates.
  for (let i = 0; i < 6; i += 1) {
    const base = pick(i * 3);
    out.push({ ...base, meta: { ...base.meta, injected: 'exact_duplicate' } });
  }
  // Cosmetic re-submissions: different casing and spacing, identical content.
  // These collapse onto the same canonical key, so the exact path catches them.
  for (let i = 0; i < 4; i += 1) {
    const base = pick(i * 5 + 1);
    out.push({
      prompt: base.prompt.replace('subject: ', 'subject:  ').toUpperCase(),
      completion: base.completion,
      meta: { ...base.meta, injected: 'cosmetic_duplicate' },
    });
  }
  // A long record re-submitted with a typo. The canonical keys differ, so only
  // the MinHash/Jaccard path can catch this one.
  const longRecord =
    'the finance team reviewed the billing refund escalation this week and the '
    + 'compliance team documented the legal audit before the renewal handoff';
  out.push({ text: longRecord, meta: { injected: 'long_record' } });
  out.push({ text: longRecord.replace('documented', 'documentd'), meta: { injected: 'near_duplicate' } });
  // Records carrying sensitive data.
  const piiSamples = [
    'reach the owner at ops.lead@contoso.example for the billing refund urgent case',
    'customer phone 13800138000 asked about the shipping invoice routine handoff',
    'service token sk-abcdefghijklmnopqrstuvwx must be rotated before the access audit soon',
    'card 4111111111111111 was declined during the billing renewal critical escalation',
  ];
  for (const sample of piiSamples) {
    out.push({ prompt: renderTaskPrompt(sample), completion: renderTaskCompletion('finance', 'p3'), meta: { injected: 'pii' } });
  }
  // Junk records: too short, repetitive, placeholder.
  out.push({ text: 'ok', meta: { injected: 'too_short' } });
  out.push({ text: 'ab'.repeat(40), meta: { injected: 'repetitive' } });
  out.push({ text: 'lorem ipsum dolor sit amet placeholder body', meta: { injected: 'placeholder' } });
  return rng.shuffled(out);
}

export async function writeSyntheticCorpus(dir, options = {}) {
  const {
    taskCount = 240,
    pretrainCount = 160,
    seed = 'tmf-demo',
    includeNoise = true,
  } = options;
  await ensureDir(dir);
  const task = generateTaskRecords(taskCount, `${seed}:task`);
  const pretrain = generatePretrainDocs(pretrainCount, `${seed}:pretrain`);
  const noise = includeNoise ? generateNoisyRecords(task, `${seed}:noise`) : [];

  const taskFile = path.join(dir, 'tickets.jsonl');
  const pretrainFile = path.join(dir, 'corpus.jsonl');
  await writeJsonl(taskFile, [...task, ...noise]);
  await writeJsonl(pretrainFile, pretrain);

  return {
    dir,
    seed,
    files: { task: taskFile, pretrain: pretrainFile },
    counts: { task: task.length, pretrain: pretrain.length, noise: noise.length },
    datasets: [
      {
        name: 'support-tickets-synthetic',
        path: taskFile,
        format: 'jsonl',
        license: 'CC0-1.0',
        owner: 'taskforce-model-foundry/synthetic',
        classification: 'internal',
        collectedAt: null,
        consentRef: 'synthetic-no-consent-required',
        tags: ['synthetic', 'routing'],
      },
    ],
    pretrainDatasets: [
      {
        name: 'enterprise-prose-synthetic',
        path: pretrainFile,
        format: 'jsonl',
        license: 'CC0-1.0',
        owner: 'taskforce-model-foundry/synthetic',
        classification: 'internal',
        consentRef: 'synthetic-no-consent-required',
        tags: ['synthetic', 'pretrain'],
      },
    ],
  };
}
