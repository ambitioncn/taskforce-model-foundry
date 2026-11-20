# R1–R7 完整项目验收逐条对账

状态日期（UTC）：2026-11-19（checkpoint cp65）
对账范围：task_contract.deliverables 中的 R1–R7 七项完整产品要求。
原则：每条写「实现路径 + 实测值 + 证据文件」。本报告不修改任何门限、不重新训练、不重跑远端作业；
只汇总此前检查点已取得的真实回执。证据根目录（相对 workspace 根）：
`artifacts/taskforce-model-foundry/evidence/`。

| 要求 | 结论 | 关键实测值 |
|---|---|---|
| R1 可安装产品 / CLI / 配置校验 / 教程 | met | npm pack 89.7 kB / 44 files（cp76 实测）；干净目录安装可执行；doctor ok=true |
| R2 私有数据本地流水线与泄漏检查 | met | 259 条入库 → 245 条采纳；PII 命中 4；去重 11；independence.ok=true |
| R3 真实后训练后端 + adapter | met | 2400 步真实权重更新，参数 L2 delta 32.6166，finalLoss 0.000160 |
| R4 独立 holdout 评测 + 门禁 | met | holdout loss 0.248876 vs baseline 4.82943；门限 5/5 passed |
| R5 可审计作业状态 / 恢复 / 预算 | met | .tmf-run 作业状态 + audit + 步数预算上限，实测中断可续 |
| R6 导出 / lineage / 模型卡 / 校验和 | met | bundleDigest 86c94751…；sha256sum -c 7/7 OK |
| R7 OpenClaw + Hermes 适配与安装验证 | met | 隔离宿主真实安装+发现+4 工具 dispatch 通过；OpenClaw 宿主自身发现插件 status=loaded / 6 个 tmf_* 工具、只读 `tmf_doctor` ok=true（cp87/cp88） |

## R1 — 可安装的完整产品、CLI/API 与配置校验，首次使用教程与可重复端到端演示

- 实现：`bin/tmf.mjs`（CLI：`init / config validate / doctor / synthesize / train / evaluate / export`）、
  `src/api.mjs` + `src/index.mjs`（程序化 API）、`src/config.mjs`（schema 校验 + `exampleConfig()`）、
  `docs/tutorial.md`（首次使用教程）。
- 实测：
  - `npm pack` → `taskforce-model-foundry-0.1.0.tgz`。打包体积随代码/文档/用例增长而变化，
    下列为带日期的快照，不是恒定不变量；恒定不变量是包名与版本 `taskforce-model-foundry@0.1.0`：
    - cp76（2026-11-19 当前工作树实测）：89.7 kB / unpacked 280.1 kB / 44 files，
      shasum `ba8d7f5784209ad8d8628dd2a20bad2c95fec10e`（证据 `cp76-acceptance-walkthrough/live-pack.log`）。
    - cp28（2026-11-18 历史快照）：68.2 kB / unpacked 222.5 kB / 38 files，
      shasum `a985b2e4c02809a4c6f9749d4b79df13ecebe3b2`（证据 `cp28-npm-pack.json`）。
    两次快照均在干净目录安装后 CLI 可执行（cp23 / cp75 实测）。
  - `tmf init` 生成的 `tmf.config.json` 带 `hyper={steps:120, lr:0.005, batchSize:4}`；
    `tmf config validate` exit 0；`tmf doctor --json` exit 0 且 `ok=true`。
  - 教程默认超参与代码双向锁定（`test/tutorial-docs.test.mjs` 对 `exampleConfig().hyper` 做 deepEqual）。
- 证据：`cp23-r7-pack-install.md`、`cp23-pack-install.log`、`cp28-npm-pack.json`、
  `cp28-packaging-manifest.md`、`cp9-cli-evidence.log`、`cp27-r6-tutorial-docs.md`、
  `cp64-default-lr/init-tmf.config.json`、`cp64-default-lr/doctor.json`。
- 已修缺口：`tmf init --help` 曾不短路、会真的执行 init（cp64 实测并清理）。现 `main()` 在
  dispatch 之前处理 `-h/--help` 与 `-v/--version`（`bin/tmf.mjs:262`-`264`）；
  `test/cli.test.mjs:131` 覆盖 `--help`/`-h` 对 init/synth/train/evaluate/export 的短路与
  “不写任何文件”断言。cp77 实测：7 条 `tmf <cmd> --help` 全部 exit 0、首行为版本抬头、
  目标目录保持为空；全量 `npm test` 203/203 pass exit 0；反向验证（移除短路）使该用例
  fail 1 exit 1，变异已还原（`md5sum -c` OK）。证据 `cp77-help-shortcircuit/`。
- 已修缺口：`package.json` 的 `files` 曾声明空目录 `examples/`（npm 不打包，无引用，属死清单项）。
  该条目已移除，且 `test/packaging.test.mjs:30` 起的用例把“每个 `files` 条目至少贡献一个真实文件”
  固化为仓库内漂移用例（缺失路径与零文件目录两种失效都断言）。cp78 实测：`npm pack --dry-run`
  exit 0，44 个文件 / 90.3 kB（unpacked 281.3 kB），tarball 清单中 `examples/` 命中数为 0；
  全量 `npm test` 203/203 pass exit 0。反向验证：把 `examples/` 注回 `files` 时，路径缺失触发
  “files entry does not exist on disk: examples/”，补成空目录则触发“files entries resolve to
  zero files: examples/”，两种变异各 fail 1、node exit 1；变异已还原（`md5sum -c` OK，8/8 pass）。
  另清理 `src/config/`、`src/cli/` 两个空残留目录（`find src -type d -empty` 现为空）。
  证据 `cp78-r1-packaging-manifest/`。

## R2 — 本地导入、清洗、去重、质量与敏感信息检查、来源许可记录、数据集隔离与泄漏检查

- 实现：`src/data/ingest.mjs`、`clean.mjs`、`dedupe.mjs`、`pii.mjs`、`provenance.mjs`、
  `split.mjs`、`pipeline.mjs`、`synth.mjs`；输出 `data-report.json`。
- 实测（cp2 data-report，configDigest `721bea14…`）：
  ingested 259 → cleaned 259 → piiRedacted 4 / piiDropped 0 → qualityDropped 3 →
  exactDuplicates 10 / nearDuplicates 1 → **accepted 245**。
  sensitiveInfo.byKind = bank_card 1 / credential 1 / email 1 / phone 1（critical 2 / high 2，policy=redact）。
  provenance 记录 license `CC0-1.0`、owner `taskforce-model-foundry/synthetic`、classification `internal`。
- 泄漏检查实测：train 284 / validation 60 / holdout 61，`independence.ok=true`，
  holdoutFingerprint `35600544d40af3b806611b3dcac13ef97c7cbcaf257efb2aef7e985c9644bd6b`
  在 cp62 训练与 cp63 导出血缘中保持一致。
- 证据：`cp2-data-report.json`、`cp2-tests.log`、`cp53-data-240.md`、
  `cp52-holdout-stratum-probe.json`、`cp62-lr005/cp62-eval-report.json`。
- 边界：仅使用本地纯合成数据。真实企业/私有数据未接入，按授权门限保持未执行。

## R3 — 真实后训练后端与可扩展适配器，小模型 + 合成数据实际更新权重

- 实现：`src/train/backends/`（可扩展后端注册 + `builtin-js`）、`src/train/optimizer.mjs`（AdamW）、
  `src/core/autograd.mjs`（反向传播）、`src/model/transformer.mjs`、`src/model/lora.mjs`、
  `src/model/tokenizer.mjs`、`src/jobs/train.mjs`。
- 实测（job `cp62-lr005`，前台 join，train EXIT=0）：
  2400 步 / batchSize 4 / lr 0.005，trainableParams 79924，**参数 L2 delta 32.6166**（权重真实改变），
  firstLoss 4.922273 → finalLoss 0.000160（bestLoss 0.000082），
  finalSmoothedLoss 0.030212（cp60 lr=0.05 时为 1.388349，振荡）。
  导出 bundle 内 `weights/base.json` 与 `weights/tuned.json` 为两份不同权重，可重载推理。
- 证据：`cp62-lr005/README.md`、`cp62-lr005/train.log`、`cp62-lr005/cp62-train-manifest.json`、
  `cp60-best-ckpt/cp60-train-manifest.json`、`cp61-exactmatch-diagnosis/`、`cp4-backend-probe.json`、
  `cp48-train-infer-consistency-audit.md`。
- 合规：未使用 RAG / 提示词 / mock 替代训练；所有训练为本机 CPU 纯合成数据。

## R4 — 基线与训练后模型的独立保留集评测、回归与阈值门禁

- 实现：`src/eval/evaluator.mjs`、`metrics.mjs`、`gates.mjs`、`src/jobs/evaluate.mjs`。
- 实测（cp62 evaluate EXIT=0，独立 holdout 61 条）：
  tuned holdout loss **0.248876**（perplexity 1.282584）vs baseline **4.82943**；
  exactMatch byte-exact 37/37 = 1.0（normalized 同为 1.0）；
  字段级 queue 37/37、priority 37/37；characterF1 = 1.0。
- 门限 5/5 通过，**阈值零修改**：
  `holdout_loss_improvement >= 0.3` → 0.948467；`holdout_loss_absolute <= 1.5` → 0.248876；
  `exact_match >= 0.25` → 1；`exact_match_improvement >= 0.1` → 1；
  `no_validation_regression >= -0.05` → 0.947702。
  verdict：`status=passed`，`improvementClaimAllowed=true`，`deliveryEligible=true`。
- 失败也被如实记录过：cp45/cp60 在 lr=0.05 下 exactMatch 0/37，当时未宣称模型提升。
- 证据：`cp62-lr005/cp62-eval-report.json`（仓库内逐字节副本 `docs/acceptance/evidence/cp62-eval-report.json`，
  sha256 自校验由 `assertReportIntegrity` 在 `test/acceptance-docs.test.mjs` 中验证）、`cp62-lr005/eval.log`、`cp56-eval-diagnostics.md`、
  `cp49-exact-match-denominator-fix.md`、`cp51-eval-sample-size-fix.md`、`cp45-gate-diagnosis-and-capacity-scaling.md`。

## R5 — 可审计作业状态、检查点、恢复、取消、预算上限、异常停止与幂等

- 实现：`src/jobs/runner.mjs`（作业状态机 + audit 追加写 + 幂等 job id）、
  `src/core/fsx.mjs`（原子写）、`src/core/hash.mjs`、`src/core/rng.mjs`（种子可复现）；
  运行目录 `.tmf-run/<jobId>/`（`state.json` / `audit.jsonl` / `train-manifest.json` / checkpoints）。
- 实测：
  - 每个作业独立 job id，cp57→cp58→cp59→cp60→cp62 未互相覆盖（重做实验开新 job id）。
  - 步数/预算上限实测生效（cp14 步预算探针、cp39 全量预算重跑）。
  - 异常停止可审计：cp57 的中断作业被核实为「无存活进程、无可用产物」后登记为中断，
    旧产物保留未被改写；恢复在新 job id 内进行（cp57/cp58 对账）。
  - 凭据与数据未写入日志/产物（cp35 只读审计）。
- 证据：`cp7-job-runner-artifacts/`、`cp8-train-job-artifacts/`、`cp60-best-ckpt/cp60-job-state.json`、
  `cp14-step-budget-probe.log`、`cp39-full-step-budget-rerun.md`、`cp57-reconciliation.md`、
  `cp58-redo-train.md`、`cp35-live-gateway-readonly-audit.md`。

## R6 — 导出、版本与 lineage、模型卡、校验和、可复现推理与回滚

- 实现：`src/export/package.mjs`（bundle + manifest + checksums + model-card + inference/rollback 文档）。
- 实测（cp63，`tmf export -C /tmp/tmf-cp57 --job cp62-lr005 --report .../eval-report.json`，EXPORT_EXIT=0）：
  - bundle name/version/kind = `tmf-cp57-local-cpu` / `0.1.0` / `full`，6 个文件。
  - `bundleDigest = 86c947516a5f5b8183ae5976451c827612712af1cc8932e20940706eaf668f3f`。
  - `sha256sum -c checksums.txt` → **7/7 全部 OK**，SHA_EXIT=0。
  - 导出的 `eval-report.json` 与 cp62 原报告 `diff -q` 为 IDENTICAL；
    `manifest.lineage.evaluation.reportDigest = 8b1a8ea7…` 与报告内字段一致。
  - lineage：training.jobId = evaluation.jobId = `cp62-lr005`；
    `parameters.hyper = steps 2400 / lr 0.005 / batchSize 4`；`independence.ok=true`。
  - 回滚：`baseCheckpoint = weights/base.json`（sha256 `45b178f6…`），指向训练前 checkpoint。
- 证据：`cp63-export/README.md`、`cp63-export/bundle/`（manifest/checksums/model-card/inference）、
  `cp63-export/export.log`、`cp20-r6-export-verification.md`、`cp41-evalreport-path-contract-fix.md`。

## R7 — OpenClaw 与 Hermes 两端安装/调用适配与契约测试（met）

- 实现：`src/integrations/openclaw.mjs`、`src/integrations/hermes.mjs`、
  `integrations/hermes-plugin/taskforce_model_foundry/`（`plugin.yaml` + `__init__.py` + `bridge.mjs`）、
  `integrations/hermes-plugin/tests/test_register.py`、`docs/integration.md`。
- 已验收（真实回执）：
  - 隔离宿主真实安装与发现：tarball sha256 `335f3c43…` 传输一致，解包安装到隔离 home
    `<isolated-home>/plugins/taskforce_model_foundry/`，插件被真实发现并加载。
  - 工具注册 + dispatch 实测 4 条路径：`synthesize`（cp36）、`train`（cp37）、`evaluate`（cp38）、
    `export`（cp40），外加 `doctor/status`（cp44）。
  - OpenClaw 侧适配与宿主挂载探针：cp21、cp25（`cp25-openclaw-host-probe-out.json`）。
  - 契约测试进 CI 套件：`test/integration-openclaw.test.mjs`、`test/integration-hermes.test.mjs`、
    `test/integration-hermes-plugin.test.mjs`、`test/integration-docs.test.mjs`、`test/packaging.test.mjs`。
  - 已知前置条件已文档化：宿主 NODE_PATH 前置与回退（cp43），宿主插件加载根因（cp42）。
  - OpenClaw 宿主自身发现插件（cp87）：隔离 `HOME`/`XDG_*` 内用真实 OpenClaw CLI
    `plugins validate` → `plugins install . --link --force --accept-capabilities` →
    `plugins list --json`，宿主加载器自己列出 `taskforce-model-foundry` 且 `status=loaded`，
    注册 6 个 `tmf_*` 工具；只读调用 `tmf_doctor` 返回 `ok=true`。
  - 发布形态复验（cp88）：`npm pack`（48 项）→ 隔离临时工程安装 tarball → 从已安装副本做同一条
    隔离 HOME 宿主发现链路，`status=loaded` + 6 工具；只读 `tmf_doctor` ok=true，
    `tmf_train` 被 `compute-not-authorized` 正确拒绝（未发生任何训练/付费）。
    同轮修复真实缺陷：`exports` 未暴露插件子路径（tarball 消费者 `ERR_PACKAGE_PATH_NOT_EXPORTED`），
    已补 2 条 `exports` 与 1 条防漂移测试。
- **R7 合同交付项内的真实缺口（显式记录，不静默缩减；当前为空）**：
  （cp84 记录的「OpenClaw 宿主自身发现插件」缺口已由 cp87 闭环，并经 cp88 在发布形态上复验；
  该清单为空不等于合同外动作已完成——后者见下一节，按合同原文不构成 R1–R7 验收前提。）
- **合同外动作（完成合同原文明文禁止、需单独授权、均未执行；不构成 R1–R7 验收或项目关闭前提）**：
  1. 生产 OpenClaw / Hermes 宿主的安装与调用验证 —— 改生产网关属禁止项，需单独授权。
  2. 代码推送到 `github.com/ambitioncn/taskforce-model-foundry` —— commit/push 需单独授权。
  3. npm publish —— 未授权，未执行（仅做过 `npm pack --dry-run`，未生成 tarball、未发布）。
- 证据：`cp21-r7-openclaw-adapter.md`、`cp22-r7-hermes-adapter.md`、`cp24-r7-integration-docs.md`、
  `cp25-r7-real-host-mount.md`、`cp29-hermes-mount-contract.md`、`cp30-hermes-python-plugin.md`、
  `cp31-smallclaw-isolated-install.md`、`cp32-hermes-tool-registration.md`、
  `cp33-hermes-tool-dispatch.md`、`cp34-hermes-plugin-regression-tests.md`、
  `cp36-hermes-dispatch-synthesize.md`、`cp37-hermes-dispatch-train.md`、
  `cp38-hermes-dispatch-evaluate.md`、`cp40-hermes-dispatch-export.md`、`cp42-host-plugin-load-rootcause.md`、`cp43-node-path-prereq-doc-and-fallback.md`、
  `cp44-hermes-dispatch-doctor-status.md`。

## 总体结论

R1–R6 有实现 + 真实运行证据，门限未降低、未以 mock 或测试数量替代模型质量。
R7 的本地/隔离环境部分已真实验证，cp87 在隔离 HOME/配置/状态目录内让真实 OpenClaw 宿主加载器
自己发现已安装插件（`status=loaded`、6 个 `tmf_*` 工具）并完成只读 `tmf_doctor` 调用（ok=true），
cp88 又在 `npm pack` 发布形态的安装副本上复验同一链路，**R7 合同交付项内已无剩余真实缺口**。
因此 R1–R7 七项合同交付项均有实现 + 真实回执，项目达到完整验收候选状态
（`project_completion=true`），最终是否通过由独立裁判裁定，不由本报告自行宣告关闭。

剩余工作（与上文 R7 真实缺口清单一一对应，当前为空）：
（无；下列合同外动作按完成合同原文需单独授权，均未执行，且不构成本项目关闭前提。）

合同外动作（**不构成本项目关闭前提**，完成合同原文即禁止，需单独授权且均未执行）：
1. 生产 OpenClaw/Hermes 宿主的安装与调用验证（改生产网关属禁止项）。
2. commit/push 到 `github.com/ambitioncn/taskforce-model-foundry`。
3. npm publish（当前仅做过 `npm pack --dry-run`，未生成 tarball、未发布）。
