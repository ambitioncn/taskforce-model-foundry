// Sensitive-information detection and redaction.
//
// The foundry is pointed at private enterprise data, so every record passes
// through here before it can reach a training set. Detection is conservative
// (prefer a false positive over leaking a secret) and redaction is in-place so
// a usable record survives when only part of it is sensitive.

const LUHN_CANDIDATE = /\b(?:\d[ -]?){12,18}\d\b/g;

function luhnValid(digits) {
  let sum = 0;
  let double = false;
  for (let i = digits.length - 1; i >= 0; i -= 1) {
    let d = digits.charCodeAt(i) - 48;
    if (double) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
    double = !double;
  }
  return digits.length >= 13 && sum % 10 === 0;
}

function cnIdChecksum(id) {
  if (!/^\d{17}[\dXx]$/.test(id)) return false;
  const weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2];
  const checks = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2'];
  let sum = 0;
  for (let i = 0; i < 17; i += 1) sum += (id.charCodeAt(i) - 48) * weights[i];
  return checks[sum % 11] === id[17].toUpperCase();
}

export const PII_RULES = [
  {
    kind: 'email',
    severity: 'high',
    placeholder: '[REDACTED:EMAIL]',
    pattern: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g,
  },
  {
    kind: 'credential',
    severity: 'critical',
    placeholder: '[REDACTED:CREDENTIAL]',
    // Common secret shapes: provider keys plus generic assignments.
    pattern: /\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,}|xox[baprs]-[A-Za-z0-9-]{10,})\b/g,
  },
  {
    kind: 'credential',
    severity: 'critical',
    placeholder: '$1=[REDACTED:CREDENTIAL]',
    pattern: /\b(password|passwd|secret|api[_-]?key|access[_-]?token|private[_-]?key)\s*[=:]\s*["']?[^\s"',;]{6,}["']?/gi,
    replace: (match, name) => `${name}=[REDACTED:CREDENTIAL]`,
  },
  {
    kind: 'cn_id',
    severity: 'critical',
    placeholder: '[REDACTED:CN_ID]',
    pattern: /\b\d{17}[\dXx]\b/g,
    validate: (m) => cnIdChecksum(m),
  },
  {
    kind: 'bank_card',
    severity: 'critical',
    placeholder: '[REDACTED:CARD]',
    pattern: LUHN_CANDIDATE,
    validate: (m) => luhnValid(m.replace(/[^\d]/g, '')),
  },
  {
    kind: 'phone',
    severity: 'high',
    placeholder: '[REDACTED:PHONE]',
    pattern: /(?:\+\d{1,3}[\s-]?)?\b1[3-9]\d{9}\b|\+\d{1,3}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{4}\b/g,
  },
  {
    kind: 'ip',
    severity: 'medium',
    placeholder: '[REDACTED:IP]',
    pattern: /\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b/g,
  },
  {
    kind: 'private_key_block',
    severity: 'critical',
    placeholder: '[REDACTED:PRIVATE_KEY]',
    pattern: /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g,
  },
];

export const SEVERITY_ORDER = { low: 0, medium: 1, high: 2, critical: 3 };

export function scanPii(text) {
  const findings = [];
  const value = String(text ?? '');
  for (const rule of PII_RULES) {
    rule.pattern.lastIndex = 0;
    let match;
    while ((match = rule.pattern.exec(value)) !== null) {
      if (match[0].length === 0) {
        rule.pattern.lastIndex += 1;
        continue;
      }
      if (rule.validate && !rule.validate(match[0])) continue;
      findings.push({ kind: rule.kind, severity: rule.severity, start: match.index, length: match[0].length });
    }
  }
  findings.sort((a, b) => a.start - b.start || b.length - a.length);
  return findings;
}

export function redactPii(text) {
  let value = String(text ?? '');
  const applied = [];
  for (const rule of PII_RULES) {
    rule.pattern.lastIndex = 0;
    value = value.replace(rule.pattern, (...args) => {
      const match = args[0];
      if (rule.validate && !rule.validate(match)) return match;
      applied.push({ kind: rule.kind, severity: rule.severity });
      return rule.replace ? rule.replace(...args) : rule.placeholder;
    });
  }
  return { text: value, applied };
}

export function highestSeverity(findings) {
  return findings.reduce((worst, f) => (SEVERITY_ORDER[f.severity] > SEVERITY_ORDER[worst ?? 'low'] ? f.severity : worst), null);
}

// Policy hook used by the pipeline: `redact` keeps the record with secrets
// removed, `drop` discards it, `fail` aborts the whole job.
export function applyPiiPolicy(text, { policy = 'redact', blockSeverity = 'critical' } = {}) {
  const findings = scanPii(text);
  if (findings.length === 0) return { action: 'keep', text: String(text ?? ''), findings };
  const worst = highestSeverity(findings);
  if (policy === 'fail' && SEVERITY_ORDER[worst] >= SEVERITY_ORDER[blockSeverity]) {
    const kinds = [...new Set(findings.map((f) => f.kind))].join(', ');
    throw new Error(`pii policy "fail": ${worst} sensitive data detected (${kinds})`);
  }
  if (policy === 'drop') return { action: 'drop', text: String(text ?? ''), findings, worst };
  const { text: redacted, applied } = redactPii(text);
  return { action: 'redact', text: redacted, findings, worst, applied };
}
