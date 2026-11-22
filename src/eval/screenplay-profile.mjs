// Frozen Chinese screenplay v1 promotion policy (existing plan §8.1).
// Percent-point metrics are expressed as points, not proportions.
export const SCREENPLAY_PROFILE = 'screenplay-zh-v1';
export const SCREENPLAY_GATES = Object.freeze([
  { id: 'rule_gain_vs_dapt', metric: 'quality.ruleGainPt', comparator: 'gte', threshold: 5, sampleMetric: 'quality.constrainedCount', minSamples: 20 },
  { id: 'scene_complete', metric: 'quality.sceneCompleteRate', comparator: 'gte', threshold: 0.95, sampleMetric: 'quality.sceneCount', minSamples: 85 },
  { id: 'broad_early_eos', metric: 'quality.broadEarlyEosRate', comparator: 'lte', threshold: 0.05, sampleMetric: 'quality.broadCount', minSamples: 35 },
  { id: 'broad_not_worse_than_base', metric: 'quality.broadEarlyEosDeltaVsBase', comparator: 'lte', threshold: 0 },
  { id: 'major_category_drop', metric: 'quality.worstCategoryDropPt', comparator: 'lte', threshold: 5, sampleMetric: 'quality.majorCategoryCount', minSamples: 8 },
  { id: 'ab_win', metric: 'quality.abMinWinRate', comparator: 'gte', threshold: 0.55, sampleMetric: 'quality.abMinJudged', minSamples: 120 },
  { id: 'ab_bootstrap_lower', metric: 'quality.abMinBootstrapLower', comparator: 'gt', threshold: 0.5 },
  { id: 'ab_judges', metric: 'quality.abJudgeCount', comparator: 'gte', threshold: 2 },
  { id: 'clean_val_vs_dapt', metric: 'quality.cleanValRegressionRelative', comparator: 'lte', threshold: 0.02 },
]);

const FIELDS = ['id', 'metric', 'comparator', 'threshold', 'sampleMetric', 'minSamples'];
export function screenplayGatesMatch(gates) {
  return Array.isArray(gates) && gates.length === SCREENPLAY_GATES.length && gates.every((gate, i) =>
    FIELDS.every((key) => gate[key] === SCREENPLAY_GATES[i][key]));
}
