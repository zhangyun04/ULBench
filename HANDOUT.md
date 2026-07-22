# ULBench Project Handout

> 这是 ULBench 重构工作的项目入口文件。  
> 开始任何代码、实验或论文任务前，先读本文件；详细任务和验收标准见本地 [`roadmap.md`](../../outputs/roadmap.md)。  
> 当前基线：`main@47a6376`  
> 当前阶段：**M0 — 冻结 benchmark definition 与 claim boundary**  
> 目标投稿：AAAI

---

## 1. North Star

### Working title

**Beyond Learn-Then-Forget: ULBench, a Method-Agnostic Benchmark for Pretrained Visual Concept Unlearning**

### One-sentence contribution

ULBench evaluates whether visual concepts already known by an off-the-shelf VLM remain accessible after an unlearning intervention, using model-specific knowledge eligibility, multiple visual probes, matched retention tests, and a unified protocol spanning black-box, inference-time, and white-box methods.

### Core research question

> 当一个 VLM 在干预前已经能够视觉识别某个概念时，不同 unlearning 方法能否在保留相邻知识和通用能力的同时，使该概念在直接、间接和对抗性视觉查询中都不可恢复？

---

## 2. 已冻结的项目决策

以下决策默认不再反复讨论；只有新证据出现时才能修改，并且必须记录到本文件末尾的 Decision Log。

1. **ULBench 是 method-agnostic benchmark，不是 training-free/prompt unlearning method。**
2. Prompt suppression 只是 black-box behavioral baseline，不能代表 machine unlearning。
3. Benchmark 分离三个对象：
   - **Target**：要降低访问性的视觉概念。
   - **Intervention**：prompt、filter、activation steering、weight update 等方法。
   - **Evaluation**：eligibility、probes、retain/general utility、robustness 和 cost。
4. 方法按访问权限分组：
   - `R0 Black-box I/O`
   - `R1 Inference-state access`
   - `R2 White-box weights`
5. 主结果只评测模型在干预前已经知道、且依赖图像作答的概念。
6. 主要结论是 **concept-level behavioral accessibility**，不是训练样本 membership 或法律意义上的数据删除证明。
7. MCQ 不能单独支撑遗忘结论；必须同时有 visual controls 和至少一种非 MCQ probe。
8. Invalid output、拒答和格式失败必须单独报告，不能自动计作成功遗忘。
9. Oracle prompt 只是 instruction-following control，不进入 unlearning method 排名。
10. 在 construct-validity pilot 通过前，不启动大规模 model sweep。

---

## 3. Claim Boundary

### 可以写进论文的 claim

- ULBench 评测 off-the-shelf VLM 已表现出的视觉概念知识。
- Model-specific eligibility 排除模型原本不知道或不依赖图像即可答对的概念。
- 统一协议可以比较不同 access regime 下的 effectiveness、leakage、retain damage、general utility 和 cost。
- 三状态实验可以区分 acquisition damage 与 subsequent unlearning damage。
- 开源 benchmark data construction、protocol、method interface、manifests 和 evaluation code。

### 当前不能写的 claim

- “这些 benchmark 图片一定出现在所有模型的预训练数据中。”
- “模型参数中的知识已经被真正、彻底删除。”
- “所有 synthetic/fictitious unlearning benchmarks 都无效。”
- “Fine-tuning 必然在 unlearning 开始前破坏 general capability。”
- “Prompt 后答错说明模型已经完成 unlearning。”
- “ULBench 是第一个/最全面的 benchmark。”除非经过最新文献核验。

### 统一术语

| 不再使用 | 统一改为 |
|---|---|
| training-free unlearning benchmark | method-agnostic unlearning benchmark |
| prompt unlearning | prompt-based behavioral suppression |
| truly deleted | no longer accessible under the evaluated probes |
| seen in pretraining | already known by the off-the-shelf model |
| unlearning accuracy | knowledge access / leakage / retain fidelity |
| oracle unlearning | oracle instruction-following control |

---

## 4. 当前仓库地图

### 可复用基础

- `vqa_gen/adapters/`：现有真实图像数据源 adapters。
- `vqa_gen/pipeline/build.py`：canonical sample → VQA item。
- `vqa_gen/pipeline/qc.py`：现有数据质量检查。
- `vqa_gen/tools/make_explicit_splits.py`：forget/retain 与 train/test split。
- `scripts/split_registry.yaml`：split 配置入口。
- `experiments/intext_unlearning.py`：现有模型加载、推理、prompt conditions 和 metrics。
- `scripts/run_batch_eval.sh`：批量模型运行。
- `scripts/aggregate_results.py`：当前固定 condition 的结果聚合。

### 已知技术债务

1. `experiments/intext_unlearning.py` 约 1900 行，混合 method、model、runner、prompt 和 metrics。
2. Benchmark 目前与 `UNLEARN_SOFT/MEDIUM` 等 prompt conditions 绑定。
3. 主评测主要是 4-choice MCQ。
4. Metrics 主要是 forget/retain accuracy、invalid rate 和 prediction entropy。
5. `DatasetItem` 缺少 probe、prompt variant、aliases、matched retain 和 provenance 字段。
6. Split 目前不检查 model-specific concept eligibility。
7. `scripts/split_registry.yaml` 中存在名称与参数不一致：例如 `coco_rk10_s42` 的实际 `k` 为 1。
8. README 声称支持 ImageNet，但当前 `vqa_gen/configs/` 没有对应配置。
9. `scripts/model_list.txt` 没有可靠冻结 model revision、loader 和可用性。
10. 当前没有 `tests/` 和 CI。

---

## 5. 目标代码结构

采用增量迁移：保留 `vqa_gen/adapters/`，新建干净的 benchmark core；不要一次性推倒重写。

```text
ulbench/
  schema.py                 # benchmark item/run schemas
  eligibility.py            # M0 knowledge + visual grounding gates
  runner.py                 # method-independent evaluation runner
  probes/
    controls.py             # no-image, shuffled-image, option-only
    direct.py               # MCQ, short answer, matching
    indirect.py             # attribute, superclass, function, relation
    adversarial.py          # paraphrase, alias, multi-turn, transforms
  methods/
    base.py
    noop.py
    prompt_suppression.py
    output_filter.py
  models/
    base.py
    huggingface.py
    internvl.py
    api.py
  metrics/
    access.py
    leakage.py
    retention.py
    utility.py
    statistics.py
  audits/
    oracle_controls.py
    acquisition_confound.py
configs/
  benchmarks/
  methods/
  models/
  studies/
tests/
```

兼容策略：`experiments/intext_unlearning.py` 暂时保留为 legacy wrapper；新功能进入 `ulbench/`，不要继续扩大旧脚本。

---

## 6. 当前阶段必须先完成什么

### P0-1：写 benchmark specification

创建 `docs/benchmark_spec_v1.md`，先冻结：

- benchmark inputs/outputs；
- concept、sample、probe、method、model state 的定义；
- R0/R1/R2 capability；
- eligibility 规则的候选阈值；
- P0–P3 probe taxonomy；
- core metrics；
- unsupported method–model combination 的处理方式；
- 哪些内容属于主 benchmark，哪些属于 audit/control。

**完成标准**：不了解项目的人能区分 benchmark、method 和 prompt baseline；所有计划中的代码字段能映射到 specification。

### P0-2：建立 schema 与 validation

优先实现：

- `BenchmarkItem`
- `ProbeSpec`
- `MethodSpec`
- `ModelCapabilities`
- `RunManifest`
- 旧 JSONL migration/validator

**完成标准**：旧数据可显式迁移；缺字段、错误 `k`、split overlap、answer-position imbalance 会 fail loudly。

### P0-3：实现 visual-grounding controls

最小集合：

- `normal_image`
- `no_image`
- `shuffled_image`
- `option_only`

**完成标准**：同一 item 的四种 condition 使用统一 ID 和结果 schema，可以逐项配对统计。

### P0-4：实现 direct dual-format probes

- 保留 MCQ，但随机化并平衡 answer position。
- 新增 short answer，支持 canonical name、aliases 和 normalized matching。
- 后续再加入 semantic scorer；在人工一致性验证前不能作为唯一评分。

**完成标准**：MCQ shortcut 与 free-form knowledge access 可以分别观测。

### P0-5：实现 eligibility manifest

对于每个 `model × concept`：

- 正常图像准确率达到 `τ_acc`；
- 正常图像相对 no-image/option-only/shuffled-image 的 gap 达到 `τ_visual`；
- 跨 prompt/order variants 足够稳定；
- 样本量和 bootstrap lower bound 满足预注册规则。

输出 `eligible_concepts/<model_revision>.json`，包含通过状态、coverage 和失败原因。

**完成标准**：所有进入主 unlearning 结果的概念都能证明“模型干预前知道且依赖图像”。

---

## 7. Pilot Gate G0

在 full experiments 前，只做：

- 2 个 open-weight VLM；
- 3 个 concept axes；
- 每个 axis 3–5 个概念；
- 每个概念至少 50 个正常图像样本；
- normal/no-image/shuffled/option-only；
- MCQ + short answer + 1 个 indirect probe；
- no-op + 当前 prompt suppression；
- prompt paraphrase 与 option-order robustness。

### G0 通过条件

- 有足够候选概念通过 knowledge + visual-grounding eligibility。
- No-image、shuffled-image 和 option-only 明显低于正常图像。
- MCQ 与 short answer 对 baseline knowledge 的判断大体一致。
- Prompt/order 变化没有不可解释的大幅波动。
- Prompt suppression 在 indirect/adversarial probes 中存在可测泄漏，证明 multi-probe 设计确有必要。
- 人工检查不少于 200 个 items；free-form scorer 与人工判断的一致率至少 90%。

若 G0 失败：缩小 concept axes、修 probes 或调整任务定义。不要用更多模型掩盖 construct-validity 问题。

---

## 8. Pilot 之后的方法与实验

### 最小方法矩阵

| Regime | 最小 baseline |
|---|---|
| R0 Black-box | no-op、held-out tuned prompt suppression、ICL/self-critique、semantic output filter |
| R1 Inference-state | representation/activation steering、适用 probe 上的 decoding/logit suppression |
| R2 White-box | Gradient Ascent、NPO 或等价方法、至少一种 multimodal-specific method |

Prompt/output filter 的定位是 behavioral controls；主文不能把它们和 weight unlearning 混成同一语义。

### 两层实验

**Depth study**：3 个不同 family 的 open-weight VLM，完整 probes 与 R0/R1/R2 方法，用于回答方法比较问题。

**Breadth study**：10–13 个 instruct/thinking、dense/MoE、不同规模模型，只运行 no-op、少量 R0 baselines 和核心 probes，用于模型生态结论。

如果没有至少 2 个 closed frontier VLM，论文必须把 broad VLM claim 缩窄为 open-weight VLMs。

### Acquisition-confound audit

对代表性 synthetic/fictitious setup 保存：

- `M0`：off-the-shelf；
- `M_acq`：学习 benchmark-specific concepts 后；
- `M_u`：unlearning 后。

分别计算：

```text
Δ_acq     = U(M_acq) - U(M0)
Δ_unlearn = U(M_u)   - U(M_acq)
Δ_total   = U(M_u)   - U(M0)
```

没有这项实验，不能把 acquisition damage 写成已证实的论文动机。

---

## 9. 核心输出与结果合同

每个正式 run 必须保存：

```text
run_dir/
  run_manifest.json
  results.jsonl            # per-item records
  metrics.json
  logs/
  failures.jsonl
```

`run_manifest.json` 至少记录：

- code git SHA；
- dataset/split hash；
- model ID、revision、processor revision；
- model state：`M0/M_acq/M_u`；
- method ID、access regime 和 capability；
- seed、prompt bank、decoding、dtype、device；
- training steps、更新参数比例、GPU-hours；
- inference latency、peak memory、API version/date/cost。

核心 metrics：

- Knowledge Access；
- Worst-Case Leakage；
- Forgetting Effect relative to `M0`；
- Matched Retain Fidelity；
- Neighborhood Damage；
- General Utility Retention；
- Refusal/Invalid Rate；
- Coverage；
- Compute/latency/cost；
- Paired bootstrap 95% CI。

禁止只保存 aggregate accuracy；paper tables 必须从冻结的 per-item records 自动生成。

---

## 10. 论文 Story Contract

论文结构不按数据集逐个汇报，而按 Research Questions 组织：

1. **RQ1 Construct validity**：模型原本是否知道目标概念，是否真的看图？
2. **RQ2 Method effectiveness**：不同 regime 降低多少 direct/indirect access？
3. **RQ3 Robustness**：paraphrase、alias、multi-turn、image transform 后能否恢复？
4. **RQ4 Utility trade-off**：forget success 对 matched retain 和 general utility 的代价？
5. **RQ5 Acquisition confound**：总下降有多少在 unlearning 前已经发生？
6. **RQ6 Scale/access analysis**：规模、thinking/instruct、open/closed 和权限如何影响结果？

每项实验开始前写一句：

> This experiment tests whether ______.

每项实验结束后只能写数据直接支持的结论。没有结果时保留 `[RESULT PENDING]`，不要生成合理但虚构的数字。

### Figure 1 必须表达

```text
Existing learn-then-forget:
M0 → acquisition → M_acq → unlearning → M_u
       Δ_acq                    Δ_unlearn

ULBench:
off-the-shelf model → eligibility → method plug-in → multi-probe evaluation
```

Prompt 只能作为 R0 插件出现，不能位于 Figure 1 的中心。

---

## 11. 每次开始工作时

1. 阅读本文件与 roadmap 对应任务。
2. 运行 `git status --short`，不要覆盖已有用户修改。
3. 用一句话写明当前任务支持哪个 paper claim 或 reviewer concern。
4. 确认任务属于 P0/P1/P2；默认只推进最高优先级未完成项。
5. 修改前先定位现有 call path、schema 和兼容入口。
6. 先写或更新最小测试，再做核心改动。
7. 小样本 smoke test 通过后，才运行 GPU experiment。

推荐任务说明模板：

```markdown
Task:
Paper claim/reviewer concern:
Inputs:
Expected outputs:
Acceptance criteria:
Out of scope:
```

---

## 12. 每次结束工作时

必须留下：

- 改了什么；
- 为什么这项改动支持 benchmark construct 或 paper claim；
- 跑了哪些测试/实验；
- 结果路径和 manifest；
- 失败或未验证内容；
- 下一项明确任务；
- 是否需要修改 claim、roadmap 或 benchmark specification。

完成任务的最低 Definition of Done：

- 代码和配置通过对应 tests；
- 输出 schema 可验证；
- 随机性和版本信息已记录；
- README/spec 与行为一致；
- 没有把 invalid/refusal 静默计作 forgetting；
- 没有引入无法追溯来源的 paper 数字；
- 没有添加未经核验的 citation/BibTeX。

---

## 13. 当前建议的第一个工作任务

不要先拆 1900 行 runner。第一个任务是：

> **创建 `docs/benchmark_spec_v1.md`，冻结 benchmark object、access regimes、probe taxonomy、eligibility、metrics 和 claim boundary。**

原因：如果 specification 没有冻结，代码接口会随着 storytelling 变化而反复返工。

随后按顺序执行：

1. 修复 split registry 命名/`k` 与 README/ImageNet 不一致。
2. 新建 schema 和 validator。
3. 抽离 model/method interfaces。
4. 实现 visual controls。
5. 实现 MCQ randomization + short answer。
6. 实现 eligibility manifest。
7. 运行 2-model construct-validity pilot。

---

## 14. Decision Log

| Date | Decision | Evidence | Consequence |
|---|---|---|---|
| 2026-07-20 | 将论文从 training-free prompt benchmark 改为 method-agnostic benchmark | 拒稿 review 指出 prompt suppression、instruction following 与 unlearning 混淆 | Prompt 变为 R0 baseline；benchmark core 与 method 解耦 |
| 2026-07-20 | 先做 eligibility/construct-validity，再做 full sweep | 当前 split 不验证模型是否已知目标概念，且评测主要为 MCQ | Full experiments 受 G0 gate 约束 |
| 2026-07-22 | Thinking checkpoint 主实验固定 thinking disabled + choice_logprob 评分，结果标 "Thinking checkpoint (thinking disabled)"，不得作为 thinking-mode 性能 | contract test：`enable_thinking=False` 对 Qwen3-VL-2B-Thinking 无效（模板仍以 `<think>\n` 结尾），旧 `run_logit_thinking`（32-token）logit 0/10 命中；关闭空 think 块后 choice_logprob 恢复 10/10、与 generate 一致 10/10 | `_apply_chat_template` 强制关闭 think 块并 fail loudly；`process_split` 绕过 run_logit_thinking；新增 `--thinking_mode`（默认 disabled）；旧 Thinking-mode 结果作废 |

---

## 15. Work Log

| Date | Task | Status | Commit/artifact | Next step |
|---|---|---|---|---|
| 2026-07-20 | 建立项目 handout | Done | `HANDOUT.md` | 创建 `docs/benchmark_spec_v1.md` |
| 2026-07-20 | 冻结 benchmark specification v1 normative draft | Done | `docs/benchmark_spec_v1.md` | 修复 split registry 与 README/ImageNet 不一致 |
| 2026-07-20 | 修复 split registry 和 README/ImageNet 漂移 | Done | `scripts/split_registry.yaml`, `vqa_gen/configs/imagenet_identity_mvp.yaml` | 新建 schema 与 legacy migration/validator |
| 2026-07-20 | 新建 v1 schema、validator 与 legacy migration CLI | Done | `ulbench/schema.py`, `ulbench/validation.py`, `ulbench/tools/` | 抽离 model/method interfaces |
| 2026-07-21 | 抽离 `UnlearningMethod`/`ModelAdapter` 接口；UNLEARN_SOFT/MEDIUM 迁为 R0 method 插件、ORACLE_HARD/REVERSE 迁为 audit control（不在 method 序列，类型层面隔离）；capability mismatch 输出 spec §4.4 结构化 unsupported 记录；prompt 文本与 legacy 逐字一致由测试强制。HF/InternVL 适配器暂委托 legacy 推理函数，runner 落地后翻转依赖方向。未完成：runner、`configs/models/*.yaml`、真实模型 10-item contract 运行 | Done | `ulbench/types.py`, `ulbench/methods/`, `ulbench/audits/oracle_controls.py`, `ulbench/models/`, `tests/test_methods.py`, `tests/test_model_adapter.py`（39 tests OK, env `unlearnpipline`） | 实现 visual-grounding controls（P0-3） |
| 2026-07-21 | P0-3 visual controls：`ulbench/probes/controls.py`（no_image/option_only/shuffled_image；注册的确定性跨概念 derangement，记录 donor+seed，不可行即报错；pairing 级校验通过 `require_core_controls`） | Done | `ulbench/probes/controls.py`, `tests/test_controls.py` | P0-4 direct probes |
| 2026-07-21 | P0-4 direct probes：MCQ 选项随机化（balancing block 内答案位置按构造均衡，permutation+seed 存 per-item）、short-answer 派生（canonical+aliases 归一化去重）、注册 scorer（mcq_exact、short_answer_alias、refusal_lexicon.v1；refusal/invalid 永不折叠为 incorrect；semantic scorer 按冻结决策不实现） | Done | `ulbench/probes/direct.py`, `ulbench/probes/scorers.py`, `tests/test_direct_probes.py` | P0-5 eligibility |
| 2026-07-21 | P0-5 eligibility：spec §9.3 预注册阈值（`EligibilityThresholds`, pilot_candidate_v1）、§9.4 manifest 构建/写出（`eligible_concepts/<revision>.json`）、六类失败码、coverage 汇总；含 seeded bootstrap 下界 gate | Done（代码；真实 manifest 待 M0 GPU runs） | `ulbench/eligibility.py`, `ulbench/metrics/statistics.py`, `tests/test_eligibility.py` | 核心 metrics |
| 2026-07-21 | 核心 metrics：response accounting（access_rate/conditional_accuracy 双视角+coverage）、WCL（含 null/contained 分解与 budget 披露）、FE_access/FE_clean（不配对即报错）、matched retain fidelity（kept/lost/gained+输出一致率）、paired bootstrap CI | Done | `ulbench/metrics/`, `tests/test_metrics.py`（全套 84 tests OK） | 真实模型 contract tests |
| 2026-07-21 | 真实模型 contract test（GPU 节点 node2x32a, 2×A6000, tmux `UL`）：`ulbench/tools/contract_test.py`；**Qwen3-VL-2B-Instruct PASS**（generate 10/10 GT、logit 10/10、两路径一致）；**InternVL3-1B PASS**（model.chat 路径 10/10；缺 einops/timm 已装并固定进 requirements.txt）；**Qwen3-VL-2B-Thinking：契约 PASS 但暴露评分构念问题**——logit 路径 0/10 命中、与 generate 仅 2/10 一致（32 个 thinking token 后模型仍在推理中，digit logits 无意义）；generate 路径 max_new_tokens=64 下推理未完成，parse 抓到的是推理文本里的第一个数字而非最终答案。**现有 3-day pipeline 中所有 Thinking 模型结果可信度存疑**，已列入待决策表 | Done | `ulbench/tools/contract_test.py`, `experiments/results/contract_tests/`（3 份报告） | 决策 thinking 评分协议；跑通其余 model_list 模型 contract tests |
| 2026-07-22 | 落实 thinking 决策：`_apply_chat_template` 关闭空 `<think>` 块 + 校验后 fail loudly；`process_split` 用 `thinking_mode`（默认 disabled）绕过 `run_logit_thinking`，全模型走 choice_logprob；metrics 写 `scoring_mode/thinking_mode/model_variant_note`；`ulbench` HuggingFaceAdapter 同步（`is_thinking`→`is_thinking_checkpoint`，新增 `thinking_mode`）。GPU 复跑 **Qwen3-VL-2B-Thinking（thinking disabled）→ logit 10/10 GT、generate 10/10、两路径一致 10/10**（此前 0/10） | Done | `experiments/intext_unlearning.py`, `ulbench/models/huggingface.py`, `tests/test_thinking_disable.py`, `tests/test_model_adapter.py`（94 tests OK）, `experiments/results/contract_tests/qwen3vl2b_thinking_disabled.json` | Thinking 模型用新协议重跑主实验；作废旧 Thinking 结果 |
