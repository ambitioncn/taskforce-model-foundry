// Source and licence bookkeeping.
//
// Every accepted record carries the dataset it came from, the file and line it
// was read from, the declared licence and data classification, and a digest of
// the raw bytes. That record is what makes the exported lineage auditable: a
// reviewer can trace any trained adapter back to the exact inputs.

export const ALLOWED_CLASSIFICATIONS = ['public', 'internal', 'confidential', 'restricted'];
export const DEFAULT_TRAINABLE_CLASSIFICATIONS = ['public', 'internal', 'confidential'];

export function normalizeDatasetDescriptor(descriptor, index = 0) {
  const name = descriptor.name ?? `dataset_${index}`;
  if (!descriptor.path) throw new Error(`dataset "${name}" is missing a path`);
  if (!descriptor.license) throw new Error(`dataset "${name}" is missing a license; provenance requires one`);
  const classification = descriptor.classification ?? 'internal';
  if (!ALLOWED_CLASSIFICATIONS.includes(classification)) {
    throw new Error(`dataset "${name}" has unknown classification "${classification}"; allowed: ${ALLOWED_CLASSIFICATIONS.join(', ')}`);
  }
  return {
    name,
    path: descriptor.path,
    format: descriptor.format ?? 'auto',
    license: descriptor.license,
    licenseUrl: descriptor.licenseUrl ?? null,
    owner: descriptor.owner ?? null,
    classification,
    collectedAt: descriptor.collectedAt ?? null,
    consentRef: descriptor.consentRef ?? null,
    promptField: descriptor.promptField ?? 'prompt',
    completionField: descriptor.completionField ?? 'completion',
    textField: descriptor.textField ?? 'text',
    tags: descriptor.tags ?? [],
  };
}

export function sourceRecord(dataset, { file, line, rawSha256 }) {
  return {
    dataset: dataset.name,
    file,
    line,
    rawSha256,
    license: dataset.license,
    licenseUrl: dataset.licenseUrl,
    owner: dataset.owner,
    classification: dataset.classification,
    collectedAt: dataset.collectedAt,
    consentRef: dataset.consentRef,
  };
}

// Aggregated licence/classification view for the model card and lineage file.
export function summarizeProvenance(records) {
  const byDataset = new Map();
  for (const record of records) {
    const key = record.source.dataset;
    if (!byDataset.has(key)) {
      byDataset.set(key, {
        dataset: key,
        license: record.source.license,
        licenseUrl: record.source.licenseUrl,
        owner: record.source.owner,
        classification: record.source.classification,
        consentRef: record.source.consentRef,
        records: 0,
        files: new Set(),
      });
    }
    const entry = byDataset.get(key);
    entry.records += 1;
    entry.files.add(record.source.file);
  }
  const datasets = [...byDataset.values()]
    .map((e) => ({ ...e, files: [...e.files].sort() }))
    .sort((a, b) => (a.dataset < b.dataset ? -1 : 1));
  const licenses = [...new Set(datasets.map((d) => d.license))].sort();
  const classifications = [...new Set(datasets.map((d) => d.classification))].sort();
  return { datasets, licenses, classifications, totalRecords: records.length };
}

export function assertTrainableClassification(records, allowed = DEFAULT_TRAINABLE_CLASSIFICATIONS) {
  const violations = records.filter((r) => !allowed.includes(r.source.classification));
  if (violations.length > 0) {
    const kinds = [...new Set(violations.map((r) => r.source.classification))].join(', ');
    throw new Error(
      `${violations.length} record(s) carry a classification not permitted for training (${kinds}); allowed: ${allowed.join(', ')}`,
    );
  }
}
