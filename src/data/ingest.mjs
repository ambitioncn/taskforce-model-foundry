// Local-only ingestion of private data.
//
// Nothing here reaches the network: a dataset is a path on this machine. The
// loader understands JSONL, plain text and CSV, and attaches provenance to each
// record as it is read.

import { readFile, stat, readdir } from 'node:fs/promises';
import path from 'node:path';
import { sha256 } from '../core/hash.mjs';
import { normalizeDatasetDescriptor, sourceRecord } from './provenance.mjs';

const TEXT_EXTENSIONS = new Set(['.txt', '.md']);
const JSONL_EXTENSIONS = new Set(['.jsonl', '.ndjson']);
const CSV_EXTENSIONS = new Set(['.csv', '.tsv']);

function detectFormat(file, declared) {
  if (declared && declared !== 'auto') return declared;
  const ext = path.extname(file).toLowerCase();
  if (JSONL_EXTENSIONS.has(ext)) return 'jsonl';
  if (CSV_EXTENSIONS.has(ext)) return 'csv';
  if (TEXT_EXTENSIONS.has(ext)) return 'text';
  if (ext === '.json') return 'json';
  throw new Error(`cannot infer format for "${file}"; set dataset.format explicitly`);
}

// Minimal RFC4180-ish parser: quoted fields, doubled quotes, embedded newlines.
export function parseDelimited(content, delimiter) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < content.length; i += 1) {
    const ch = content[i];
    if (quoted) {
      if (ch === '"') {
        if (content[i + 1] === '"') {
          field += '"';
          i += 1;
        } else quoted = false;
      } else field += ch;
      continue;
    }
    if (ch === '"') quoted = true;
    else if (ch === delimiter) {
      row.push(field);
      field = '';
    } else if (ch === '\n') {
      row.push(field);
      field = '';
      rows.push(row);
      row = [];
    } else if (ch !== '\r') field += ch;
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((r) => r.some((c) => c.trim() !== ''));
}

async function listFiles(target) {
  const info = await stat(target);
  if (info.isFile()) return [target];
  const entries = await readdir(target, { withFileTypes: true });
  const out = [];
  for (const entry of entries.sort((a, b) => (a.name < b.name ? -1 : 1))) {
    const full = path.join(target, entry.name);
    if (entry.isDirectory()) out.push(...(await listFiles(full)));
    else if (entry.isFile() && !entry.name.startsWith('.')) out.push(full);
  }
  return out;
}

function pickFields(obj, dataset) {
  const prompt = obj[dataset.promptField];
  const completion = obj[dataset.completionField];
  const text = obj[dataset.textField];
  if (prompt !== undefined && completion !== undefined) {
    return { prompt: String(prompt), completion: String(completion) };
  }
  if (text !== undefined) return { prompt: null, completion: null, text: String(text) };
  throw new Error(
    `record is missing usable fields; expected "${dataset.promptField}"+"${dataset.completionField}" or "${dataset.textField}"`,
  );
}

function makeRecord(dataset, fields, source) {
  const text = fields.text ?? `${fields.prompt}${fields.completion}`;
  return {
    id: sha256(`${source.dataset}:${source.file}:${source.line}:${text}`).slice(0, 16),
    prompt: fields.prompt ?? null,
    completion: fields.completion ?? null,
    text,
    source,
  };
}

export async function ingestDataset(descriptor, index = 0, { baseDir = process.cwd() } = {}) {
  const dataset = normalizeDatasetDescriptor(descriptor, index);
  const target = path.isAbsolute(dataset.path) ? dataset.path : path.join(baseDir, dataset.path);
  let files;
  try {
    files = await listFiles(target);
  } catch (err) {
    throw new Error(`dataset "${dataset.name}" path is unreadable: ${err.message}`);
  }
  const records = [];
  const errors = [];
  for (const file of files) {
    const content = await readFile(file, 'utf8');
    const format = detectFormat(file, dataset.format);
    const relative = path.relative(baseDir, file) || path.basename(file);
    if (format === 'jsonl') {
      const lines = content.split('\n');
      for (let i = 0; i < lines.length; i += 1) {
        const line = lines[i].trim();
        if (!line) continue;
        try {
          const fields = pickFields(JSON.parse(line), dataset);
          records.push(makeRecord(dataset, fields, sourceRecord(dataset, { file: relative, line: i + 1, rawSha256: sha256(line) })));
        } catch (err) {
          errors.push({ file: relative, line: i + 1, reason: err.message });
        }
      }
    } else if (format === 'json') {
      try {
        const parsed = JSON.parse(content);
        const items = Array.isArray(parsed) ? parsed : parsed.records ?? [];
        items.forEach((item, i) => {
          try {
            const fields = pickFields(item, dataset);
            const raw = JSON.stringify(item);
            records.push(makeRecord(dataset, fields, sourceRecord(dataset, { file: relative, line: i + 1, rawSha256: sha256(raw) })));
          } catch (err) {
            errors.push({ file: relative, line: i + 1, reason: err.message });
          }
        });
      } catch (err) {
        errors.push({ file: relative, line: 0, reason: `invalid json: ${err.message}` });
      }
    } else if (format === 'csv') {
      const delimiter = path.extname(file).toLowerCase() === '.tsv' ? '\t' : ',';
      const rows = parseDelimited(content, delimiter);
      if (rows.length === 0) continue;
      const header = rows[0].map((h) => h.trim());
      for (let i = 1; i < rows.length; i += 1) {
        const obj = Object.fromEntries(header.map((h, c) => [h, rows[i][c] ?? '']));
        try {
          const fields = pickFields(obj, dataset);
          records.push(makeRecord(dataset, fields, sourceRecord(dataset, { file: relative, line: i + 1, rawSha256: sha256(rows[i].join(delimiter)) })));
        } catch (err) {
          errors.push({ file: relative, line: i + 1, reason: err.message });
        }
      }
    } else {
      // Plain text: blank-line separated documents.
      const chunks = content.split(/\n\s*\n/);
      let line = 1;
      for (const chunk of chunks) {
        const value = chunk.trim();
        const lineStart = line;
        line += chunk.split('\n').length;
        if (!value) continue;
        records.push(makeRecord(dataset, { text: value }, sourceRecord(dataset, { file: relative, line: lineStart, rawSha256: sha256(value) })));
      }
    }
  }
  return { dataset, files: files.length, records, errors };
}

export async function ingestAll(descriptors, { baseDir = process.cwd() } = {}) {
  if (!Array.isArray(descriptors) || descriptors.length === 0) {
    throw new Error('at least one dataset descriptor is required');
  }
  const results = [];
  for (let i = 0; i < descriptors.length; i += 1) {
    results.push(await ingestDataset(descriptors[i], i, { baseDir }));
  }
  return {
    records: results.flatMap((r) => r.records),
    errors: results.flatMap((r) => r.errors),
    datasets: results.map((r) => ({ name: r.dataset.name, files: r.files, records: r.records.length, errors: r.errors.length })),
  };
}
