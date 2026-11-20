// Filesystem helpers. Writes go through a temp file + rename so a crashed or
// cancelled job never leaves a half-written checkpoint or report behind.

import { mkdir, writeFile, rename, readFile, rm, access } from 'node:fs/promises';
import { constants } from 'node:fs';
import path from 'node:path';

export async function ensureDir(dir) {
  await mkdir(dir, { recursive: true });
  return dir;
}

export async function writeFileAtomic(file, content) {
  await ensureDir(path.dirname(file));
  const tmp = `${file}.tmp-${process.pid}-${Date.now().toString(36)}`;
  await writeFile(tmp, content);
  await rename(tmp, file);
  return file;
}

export async function writeJson(file, value, { pretty = true } = {}) {
  return writeFileAtomic(file, `${JSON.stringify(value, null, pretty ? 2 : 0)}\n`);
}

export async function readJson(file) {
  return JSON.parse(await readFile(file, 'utf8'));
}

export async function readJsonIfExists(file, fallback = null) {
  try {
    return await readJson(file);
  } catch (err) {
    if (err.code === 'ENOENT') return fallback;
    throw err;
  }
}

export async function writeJsonl(file, rows) {
  return writeFileAtomic(file, rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
}

export async function appendJsonl(file, row) {
  const { appendFile } = await import('node:fs/promises');
  await ensureDir(path.dirname(file));
  await appendFile(file, `${JSON.stringify(row)}\n`);
  return file;
}

export async function readJsonl(file) {
  const content = await readFile(file, 'utf8');
  return content.split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));
}

export async function exists(target) {
  try {
    await access(target, constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

export async function removeDir(dir) {
  await rm(dir, { recursive: true, force: true });
}
