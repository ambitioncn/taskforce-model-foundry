// Package entry point: the stable public surface.
//
// High-level operations come from ./api.mjs; the lower-level building blocks
// are re-exported under namespaces so embedders can compose their own jobs
// without reaching into file paths that may move.

export { PRODUCT, doctor, synthesize, train, status, evaluate } from './api.mjs';
export { exportModel as exportBundle } from './api.mjs';
export { qualityContext } from './api.mjs';
export {
  CONFIG_FORMAT, CONFIG_DEFAULTS, validateConfig, loadConfig, findConfigFile,
  exampleConfig, formatConfigErrors, ConfigError,
} from './config.mjs';

export * as data from './data/pipeline.mjs';
export * as synth from './data/synth.mjs';
export * as jobs from './jobs/runner.mjs';
export * as trainJob from './jobs/train.mjs';
export * as evalJob from './jobs/evaluate.mjs';
export * as gates from './eval/gates.mjs';
export * as externalQuality from './eval/external-quality.mjs';
export * as backends from './train/backends/index.mjs';
export * as exporter from './export/package.mjs';
export * as controller from './controller/index.mjs';
export { exportModel, assertDeliverable, ExportBlockedError, EXPORT_FORMAT } from './export/package.mjs';

export { VERSION } from './version.mjs';
export * as openclaw from './integrations/openclaw.mjs';
export * as hermes from './integrations/hermes.mjs';
