import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readdir, stat } from 'node:fs/promises';
import path from 'node:path';

export function sha256(data) {
  return createHash('sha256').update(data).digest('hex');
}

export function sha256Json(value) {
  return sha256(stableStringify(value));
}

// Stable key ordering so lineage digests do not drift with object literal order.
export function stableStringify(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value) ?? 'null';
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringify(value[k])}`).join(',')}}`;
}

export async function sha256File(file) {
  const hash = createHash('sha256');
  await new Promise((resolve, reject) => {
    createReadStream(file).on('data', (c) => hash.update(c)).on('end', resolve).on('error', reject);
  });
  return hash.digest('hex');
}

// Checksum manifest for an exported bundle: every regular file, relative path, sorted.
export async function checksumTree(root) {
  const entries = [];
  async function walk(dir) {
    const items = await readdir(dir, { withFileTypes: true });
    for (const item of items.sort((a, b) => (a.name < b.name ? -1 : 1))) {
      const full = path.join(dir, item.name);
      if (item.isDirectory()) await walk(full);
      else if (item.isFile()) {
        const info = await stat(full);
        entries.push({ path: path.relative(root, full), bytes: info.size, sha256: await sha256File(full) });
      }
    }
  }
  await walk(root);
  entries.sort((a, b) => (a.path < b.path ? -1 : 1));
  return entries;
}
