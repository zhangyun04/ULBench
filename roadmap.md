# ULBench → AAAI 重构 Roadmap

> 工作题目：**Beyond Learn-Then-Forget: ULBench, a Method-Agnostic Benchmark for Pretrained Visual Concept Unlearning**  
> 目标：把当前以 in-text/prompt suppression 为中心的工作，重构为一个 **method-agnostic、面向模型已具备视觉知识的 concept-level unlearning benchmark**。  
> 状态日期：2026-07-20  
> 代码基线：[`zhangyun04/ULBench`](https://github.com/zhangyun04/ULBench) `main@47a6376`  
> Review 来源：`/Users/tzy/Downloads/review.pdf`

---

## 0. 如何维护这份 Roadmap

- 状态：`[ ]` 未开始，`[~]` 进行中，`[x]` 完成，`[!]` 阻塞或需要决策。
- 优先级：
  - **P0 / Must-have**：不完成就不建议再次投稿；直接决定论文定义是否成立。
  - **P1 / Strongly recommended**：显著影响 AAAI 竞争力和 reviewer 信心。
  - **P2 / Nice-to-have**：在 P0/P1 稳定后扩展，不允许阻塞主线。
- 每完成一个任务，同时补充：负责人、commit、产物路径、关键结果和是否改变论文 claim。
- 论文中的数字只能来自冻结的结果目录和 manifest，禁止从临时日志手抄。
- 本路线图按依赖顺序排列。即使两条线路并行，也不能在核心评测定义未冻结前大规模烧算力。

---

## 1. 论文最终要讲的故事

### 1.1 一句话主张

现有多模态 unlearning 评测经常先让模型学习合成的、原本不知道的概念，再要求模型遗忘；这一 **learn-then-forget** 流程会把 acquisition/post-training 带来的能力变化与真正的 unlearning effect 混在一起。ULBench 改为评测 off-the-shelf VLM 已经表现出视觉知识的真实概念，并用跨访问权限、跨 probe 的统一协议衡量知识是否仍可被访问。

### 1.2 论文回答的核心问题

> 当一个 VLM 在干预前已经能够视觉识别某个概念时，不同 unlearning 方法能否在保留通用能力的同时，使该概念在直接、间接和对抗性视觉查询中都不可恢复？

### 1.3 贡献边界

论文可以声称：

1. 提供一个针对 **pre-existing visual concept knowledge** 的 method-agnostic benchmark。
2. 用 model-specific eligibility 筛除模型本来就不知道、或不依赖图像即可答对的概念。
3. 在统一的 access-regime 与 probe taxonomy 下比较行为抑制、推理时干预和权重修改方法。
4. 区分 acquisition degradation、unlearning degradation、retain damage 和 general-utility damage。
5. 开源数据构建、协议、方法接口、结果 manifest 和评测代码。

论文不能声称：

- 不根据训练数据 provenance，声称某张 benchmark 图片一定出现在模型预训练集里。
- 仅凭输出拒答或 MCQ 掉点，证明参数中对应信息已经被彻底删除。
- 把 system prompt、prompt refusal 或 output filter 直接称为 parameter unlearning。
- 在没有完整对照实验前，声称所有 synthetic/fictitious benchmarks 都无效。
- 用 oracle prompt 的结果作为“真正遗忘”的证据；它只能是 instruction-following positive control。

### 1.4 推荐术语

| 避免 | 改用 |
|---|---|
| training-free unlearning benchmark | method-agnostic unlearning benchmark |
| prompt unlearning | prompt-based behavioral suppression |
| truly deleted from the model | no longer behaviorally accessible under the evaluated probes |
| seen during pretraining | already known by the off-the-shelf model |
| unlearning accuracy | knowledge access / leakage / retain fidelity |
| oracle unlearning | oracle instruction-following control |

---

## 2. 当前代码基线审计

### 2.1 可以保留和复用的部分

- `[x]` `vqa_gen/adapters/` 已支持 COCO、SpatialMQA、MIT Indoor-67、AID、LAD、Celebrity Faces、Logo-2K+ 等真实图像来源。
- `[x]` `vqa_gen/pipeline/build.py` 已有 canonical sample → VQA item 的构建流程。
- `[x]` `vqa_gen/tools/make_explicit_splits.py` 已有 class-level forget/retain 与 train/test 拆分、随机 K 和 superclass-balanced K。
- `[x]` `scripts/split_registry.yaml` 已能集中管理多数据集 split。
- `[x]` `experiments/intext_unlearning.py` 已有多模型加载、batch inference、logit evaluation 和多 GPU MapReduce 基础。
- `[x]` `scripts/run_batch_eval.sh`、`04_run_experiments.sh` 和 `aggregate_results.py` 已形成基本批量实验链路。

### 2.2 必须修复的结构问题

| 当前状态 | 风险 | 对应改动 |
|---|---|---|
| `experiments/intext_unlearning.py` 约 1900 行，混合 prompt、method、model、runner、metrics | benchmark 与某一方法耦合；难以公平接入训练式方法 | 拆成 schema、model adapter、method adapter、probe、runner、metrics |
| 条件固定为 `UNLEARN_SOFT/MEDIUM`、`ORACLE_HARD/REVERSE` | reviewer 会继续认为贡献是 prompt study | 把 condition 移出 benchmark core，作为 method/config 插件 |
| 主评测以 4-choice MCQ 为主 | 可被选项语义、位置、拒答和 output suppression 混淆 | 加 visual controls、free-form、matching、indirect 和 adversarial probes |
| 当前 metrics 主要是 forget/retain accuracy、invalid rate、entropy | 无法区分 suppression、leakage、utility 和 robustness | 增加 eligibility、worst-case leakage、paired retain fidelity、utility、cost、CI |
| `DatasetItem` 缺少 probe、prompt variant、alias、matched retain、license/provenance 字段 | 难以追踪审计和可重复性 | 扩展 schema，并提供旧 JSONL migration |
| 当前 split 按样本数/类别数选概念，不检查模型是否真的知道 | 低 baseline accuracy 会制造“伪遗忘” | 先跑 model-specific eligibility，再生成 eligible concept manifest |
| `scripts/split_registry.yaml` 中部分实验名与实际 `k` 不一致，例如 `coco_rk10_s42` 的 `k: 1` | 容易误报实验设置 | schema validation + CI 检查名称/参数一致性 |
| README 声称 ImageNet 支持，但当前 `vqa_gen/configs/` 未见对应 config | 文档与可运行代码不一致 | 补 config 或从主 benchmark 清单移除，并在 release 前自动校验 |
| `scripts/model_list.txt` 是人工清单，且注释、可用性和版本冻结不完整 | 模型 ID/版本漂移，结果难复现 | 改为带 revision、dtype、template、license、status 的 YAML manifest |
| 当前没有 `tests/` 和 CI | 重构后容易出现静默指标错误 | 增加 unit/integration/schema tests 和 smoke workflow |

---

## 3. 总体优先级与关键路径

1. **P0：冻结 benchmark 定义和 claim 边界。** 先回答“测什么”，再写代码。
2. **P0：完成视觉知识 eligibility 与 confound controls。** 没有它们，pre-existing visual knowledge 的故事不成立。
3. **P0：把 benchmark core 与 method 解耦。** 让 prompt suppression 成为 baseline，而不是 benchmark 本身。
4. **P0：做小规模 construct-validity pilot。** 先证明 probes 和 metrics 测到目标变量。
5. **P0：接入每种 access regime 的最小方法集合。** 避免“method-agnostic”只停留在接口层。
6. **P0：完成 depth study、breadth study 和 acquisition-confound audit。** 这是 paper results 的证据骨架。
7. **P0：根据结果写 Abstract/Introduction/Results。** 结果未冻结前只写无数字版本。
8. **P1：补 closed models、规模分析、更多 utility benchmarks 和统计稳健性。**
9. **P2：做 leaderboard、更多隐私概念和更广攻击面。**

### Go/No-Go Gate G0：是否值得继续这个 framing

用 2 个 open VLM、每个 axis 3–5 个概念做最小 pilot。只有同时满足以下条件才进入 full run：

- 至少有足够比例的候选概念通过 image accuracy 与 image/no-image gap 门槛。
- shuffled-image、no-image 和 option-only 控制显著低于正常图像条件。
- free-form 与 MCQ 对“模型是否知道概念”的结论大体一致。
- prompt/option order 的扰动不会让 baseline 结果发生不可解释的大幅波动。
- 现有 prompt suppression 在更强 probes 下出现可测的泄漏，从而证明多 probe 设计有必要。

如果 G0 失败，先缩小 concept axes 或重做 probes，不进入大规模模型 sweep。

---

# 线路 A：论文 Storytelling 与写作重构

## A-P0.1 冻结论文定位、题目和任务定义

- `[ ]` 把工作题目固定为下列两者之一：
  - 推荐：**Beyond Learn-Then-Forget: ULBench, a Method-Agnostic Benchmark for Pretrained Visual Concept Unlearning**
  - 备选：**Can VLMs Truly Forget? A Method-Agnostic Benchmark for Pretrained Visual Concept Unlearning**
- `[ ]` 在正文第一页定义三层概念：
  1. **Target**：要降低访问性的概念知识。
  2. **Intervention**：prompt、filter、activation、weight update 等方法。
  3. **Evaluation**：eligibility、probe families、retain/utility、attack 与成本。
- `[ ]` 明确三种 access regime：
  - **R0 Black-box I/O**：只能改输入或后处理输出。
  - **R1 Inference-state access**：可访问 logits/hidden states/decoding。
  - **R2 White-box weights**：可更新、剪枝或编辑参数。
- `[ ]` 明确 benchmark 不要求所有方法共享相同权限；比较时先在 regime 内公平比较，再报告跨 regime 的 effectiveness–utility–cost Pareto frontier。

**验收标准**：一个不了解项目的人读完 150 字 task definition 后，能清楚区分 benchmark、prompt baseline 和 machine unlearning；不再出现“ULBench prescribes a prompt method”的理解。

## A-P0.2 重写 Abstract

Abstract 固定为五步，不再从方法细节开头：

1. **真实需求**：部署中的 VLM 可能需要移除已掌握的敏感、版权或受限视觉概念。
2. **评测缺口**：learn-then-forget synthetic setup 把 acquisition damage 与 unlearning damage 混淆；单一 prompt/MCQ 又把知识删除和输出抑制混淆。
3. **核心方案**：ULBench 在 off-the-shelf 模型上先验证 concept eligibility，再以 method-agnostic 协议跨 access regimes 评测。
4. **证据设计**：direct、indirect、adversarial probes + matched retain + general utility + cost + acquisition audit。
5. **主要发现**：只写 full experiment 后被数据支持的 2–3 个结论。

**写作限制**：

- 不在 abstract 中把“fine-tuning 一定先破坏能力”写成已知事实；应写成待审计的 confound，随后用三状态实验验证。
- 不在结果出来前写“first”“comprehensive”“truly forget”等强 claim。
- 不列七个数据集和十几个模型的流水账，优先写 benchmark design 带来的科学结论。

**验收标准**：150–220 词；第一遍阅读即可回答 gap、idea、protocol、finding、significance 五个问题。

## A-P0.3 重写 Introduction 为一条因果链

建议 6 段结构：

1. **Problem**：为什么视觉概念需要被移除，以及“答不出来”不等于“已经忘记”。
2. **Mismatch**：现有 synthetic/fictitious learn-then-forget 设置为什么与 off-the-shelf removal request 不完全一致。
3. **Confounds**：acquisition damage、instruction following、output suppression、MCQ shortcut 四个混淆源。
4. **Design principle**：一个可信 benchmark 必须满足 known-before-forgetting、visually grounded、method-independent、multi-probe、utility-aware。
5. **ULBench overview**：数据/概念、eligibility、access regimes、probes、metrics、audit。
6. **Findings + contributions**：只列结果支持的结论；贡献控制在 3 条。

贡献 bullets 推荐结构：

- **Problem formulation**：把 pre-existing visual concept unlearning 与 synthetic acquisition/unlearning 解耦。
- **Benchmark**：model-specific eligibility + multi-probe + access-regime-aware evaluation。
- **Empirical study**：展示不同方法在 leakage、retain、utility 和 cost 上的真实 trade-off，并量化 acquisition confound。

**验收标准**：Introduction 中每一段都有单一功能；删掉任意一段会破坏论证链，而不是只损失背景材料。

## A-P0.4 设计 Figure 1：让 reviewer 在 20 秒内理解论文

- `[ ]` 左侧画现有 learn-then-forget：`M0 → acquisition → M_acquired → unlearning → M_unlearned`。
- `[ ]` 标出 `Δ_acquisition` 与 `Δ_unlearning`，说明只比较 `M0` 和 `M_unlearned` 会归因错误。
- `[ ]` 右侧画 ULBench：off-the-shelf model → eligibility → method intervention → multi-probe evaluation。
- `[ ]` 把 method 画成可插拔模块，三种颜色对应 R0/R1/R2；prompt 只是 R0 中的一个方块。
- `[ ]` 把 no-image/shuffled-image control 放在 eligibility，而不是藏进 appendix。

**验收标准**：Figure 1 不依赖正文即可表达“旧评测的混淆”和“ULBench 的解法”；不把任何具体 prompt 放在视觉中心。

## A-P0.5 Formalization / Benchmark Section

- `[ ]` 定义 benchmark 对象：概念集合、候选样本、forget/retain split、probe set、method capability、model state 和 metrics。
- `[ ]` 定义 model-specific eligibility：
  - 正常图像表现达到绝对门槛 `τ_acc`。
  - 正常图像与 no-image/option-only 的差值达到视觉 grounding 门槛 `τ_visual`。
  - 对多个 prompt/order variants 稳定。
- `[ ]` 定义模型状态：
  - `M0`：off-the-shelf。
  - `M_acq`：仅用于 synthetic benchmark confound audit。
  - `M_u`：应用 intervention 后。
- `[ ]` 定义 probe families：
  - `P0` Visual-grounding controls。
  - `P1` Direct identification：MCQ + short answer。
  - `P2` Indirect access：attribute、superclass、function、relation、description、image-text matching。
  - `P3` Adversarial recovery：paraphrase、alias、multi-turn、instruction conflict、image transform。
- `[ ]` 解释 method-agnostic 不等于 method-unaware：benchmark 记录每个方法的权限、训练数据、更新参数量和推理预算。
- `[ ]` 明确 evaluation target 是 **concept-level behavioral accessibility**，不是训练样本 membership certification。

**验收标准**：所有论文指标都能从 formalization 中找到定义；所有代码字段都能映射到公式变量。

## A-P0.6 Results 章节按科学问题组织，不按数据集组织

建议顺序：

1. **RQ1 Construct validity**：模型是否真的在看图、是否原本知道目标概念？
2. **RQ2 Method effectiveness**：各 access regime 能降低多少直接和间接知识访问？
3. **RQ3 Robustness**：在 paraphrase、alias、multi-turn 和 image transforms 下能否恢复？
4. **RQ4 Utility trade-off**：forget 成功是否以 matched retain 或 MMMU/MMStar 等通用能力为代价？
5. **RQ5 Acquisition confound**：learn-then-forget 中总下降有多少在 unlearning 前已经发生？
6. **RQ6 Scale/access analysis**：模型规模、thinking/instruct、open/closed、访问权限如何改变 Pareto frontier？

每个小节遵循：问题 → 关键图表 → 一句话结果 → 机制/反例 → 限制。避免逐模型报数。

**验收标准**：每个 RQ 最多承载一个主结论；表格中的数字不重复写成长段 prose；所有结论带 CI 或显著性/效应量。

## A-P0.7 Related Work 用“差异轴”组织

不要简单按论文时间顺序罗列。建议四组：

1. **Language-model real-world unlearning benchmarks**：重点学习 RWKU 如何把 pretrained real-world knowledge 变成评测对象。
2. **Multimodal unlearning benchmarks**：FIUBench、CLEAR、MLLMU-Bench 等，按 real/fictitious、identity/concept、acquisition requirement、probe breadth 比较。
3. **Multimodal unlearning methods**：SIU、MANU、MMUnlearner、R-MUSE 等，按 R0/R1/R2 分类，而不是把它们都放入“baseline”长列表。
4. **Benchmark validity/control design**：借鉴 MMStar、SugarCrepe 的 image-absent、shortcut 和 compositional hard-negative 控制思路。

必须加入一张 comparison table，列：真实/合成概念、是否要求 acquisition、视觉 grounding control、method-agnostic、free-form、adversarial probes、retain/general utility、access regimes。

**验收标准**：table 中每个勾选都有原论文依据；避免“ours is the first”式无法防守的结论。

## A-P0.8 Discussion、Limitations 与 Ethics

- `[ ]` 区分 behavioral suppression、representation intervention 和 parameter modification。
- `[ ]` 说明 benchmark 不能证明训练数据级法律删除或 membership removal。
- `[ ]` 说明 concept-level unlearning 可能伤害相邻、上位或组合概念。
- `[ ]` 讨论 celebrity/logo 数据的许可、隐私和发布方式；公开 metadata/splits 与公开原图需分别审查。
- `[ ]` 讨论 closed API 版本漂移、不可复现实验和不可观测参数状态。
- `[ ]` 讨论开放式答案 judge bias，并报告人工一致性抽查。

**验收标准**：Limitations 主动覆盖 reviewer 最容易攻击的边界，但不否定论文自己的 operational value。

## A-P1 写作增强项

- `[ ]` **A-P1.1** Figure 2：benchmark pipeline 与 schema；突出 method 插件位置。
- `[ ]` **A-P1.2** Figure 3：effectiveness–retain–utility–cost Pareto，而不是只报平均准确率。
- `[ ]` **A-P1.3** Figure 4：direct/indirect/adversarial leakage 热图或雷达图。
- `[ ]` **A-P1.4** Table 1：与现有 benchmarks 的差异。
- `[ ]` **A-P1.5** Table 2：depth study 主结果；每个 regime 至少一个强 baseline。
- `[ ]` **A-P1.6** Table 3：acquisition confound 分解。
- `[ ]` **A-P1.7** Appendix：完整 prompt variants、method hyperparameters、concept eligibility manifest、license、compute 和失败案例。
- `[ ]` **A-P1.8** 给每张表写一句“takeaway title”，避免标题只描述内容。
- `[ ]` **A-P1.9** AAAI 页数压缩时优先保留 task definition、Figure 1、主表、confound audit；把模型加载和长 prompt 移到 appendix。

## A-P2 可选增强

- `[ ]` 加入 case-study：同一概念在 direct probe 已拒答，但通过 attribute/description 被恢复。
- `[ ]` 加入邻接概念知识图，展示 over-unlearning 的语义扩散。
- `[ ]` 写 benchmark card 和 model/method card 模板。
- `[ ]` 如果有稳定结论，再恢复标题中的问句 “Can VLMs Truly Forget?”；否则使用更精确的陈述式标题。

---

# 线路 B：代码重构与实验

## B-P0.1 冻结 benchmark schema 和 release manifest

建议新增：

```text
ulbench/
  schema.py
  eligibility.py
  runner.py
  probes/
  methods/
  models/
  metrics/
  audits/
configs/
  benchmarks/
  methods/
  models/
  studies/
tests/
```

保留 `vqa_gen/adapters/` 和现有构建逻辑，逐步迁移，不做一次性推倒重写。

- `[ ]` 扩展 item schema，至少包含：
  - `sample_id`, `dataset_id`, `image_id`, `concept_id`, `concept_aliases`
  - `forgetting_level`, `concept_axis`, `split`
  - `probe_family`, `question_format`, `prompt_variant_id`
  - `question`, `choices`, `accepted_answers`, `answer_index`
  - `matched_retain_id`, `source`, `license`, `provenance_note`
- `[ ]` 新增 run manifest：
  - repo git SHA、dataset version/hash、model ID + revision、processor revision
  - model state `M0/M_acq/M_u`、method ID、access regime、seed
  - decoding参数、dtype、device、更新参数量、训练/推理时长、峰值显存、API version/date
- `[ ]` 为旧 `DatasetItem` JSONL 写 migration/validation 工具。
- `[ ]` 给 YAML 加 JSON Schema 或 Pydantic validation；检查 split 名称与 `k`、路径、数据量一致。

**对应现有文件**：`vqa_gen/internal/types.py`、`vqa_gen/pipeline/build.py`、`scripts/split_registry.yaml`。

**验收标准**：同一个结果可以仅凭 manifest 在新环境重建；无字段的旧结果不能被静默混入主表。

## B-P0.2 把 method 从 runner 中解耦

定义统一接口，例如：

```python
class UnlearningMethod:
    access_regime: str
    def prepare(self, model, forget_set, retain_set, config): ...
    def transform_input(self, request): ...
    def intervene(self, model_state): ...
    def transform_output(self, response): ...
    def metadata(self) -> dict: ...
```

- `[ ]` 把 `build_prompt()` 中 `UNLEARN_SOFT/MEDIUM` 迁到 `ulbench/methods/prompt_suppression.py`。
- `[ ]` 把 `ORACLE_HARD/REVERSE` 迁到 `ulbench/audits/oracle_controls.py`，不再列入 unlearning methods 主表。
- `[ ]` runner 只接受 `model_adapter + method_adapter + probe_suite + metric_suite`。
- `[ ]` method 声明需要的能力：I/O、logits、hidden states、gradients、weight write、retain set。
- `[ ]` 不支持的 model–method 组合返回明确 capability error，而不是 runtime crash 或空结果。
- `[ ]` 保留 `experiments/intext_unlearning.py` 为兼容 wrapper，并标记 deprecated；主实验改走 `ulbench.runner`。

**验收标准**：新增一个 no-op method 和一个 prompt method 不需要改 runner；同一 probe suite 可原样用于 prompt、activation 和 weight-update 方法。

## B-P0.3 抽离 model adapter

- `[ ]` 从 `load_model_and_processor()`、`run_logit_batch()`、`run_internvl_batch()` 抽出统一 `ModelAdapter`。
- `[ ]` 支持以下能力查询：`supports_images`、`supports_logits`、`supports_hidden_states`、`supports_gradients`、`supports_system_prompt`、`is_closed_api`。
- `[ ]` 分离 instruct/thinking 模型逻辑；保存最终答案与 reasoning trace 的可见部分，但默认不把 chain-of-thought 当作可公开数据。
- `[ ]` 把 `scripts/model_list.txt` 迁移为 `configs/models/*.yaml`，冻结 model revision、chat template 和 loader type。
- `[ ]` 给每个模型跑 10-item contract test：同一输入字段、相同输出 schema、无 silent fallback。

**验收标准**：模型特殊分支不再散落在 evaluator 中；失败样本带明确 error taxonomy，invalid 不再自动等价于成功遗忘。

## B-P0.4 建立 multi-probe suite 与反捷径控制

### P0 Visual-grounding controls

- `[ ]` `normal_image`：原图。
- `[ ]` `no_image`：去掉视觉输入，保留问题与选项。
- `[ ]` `shuffled_image`：同 batch 随机错配图片。
- `[ ]` `option_only`：只给选项，检查答案位置/语义先验。
- `[ ]` `question_only`：不给选项，检查语言先验。

### P1 Direct probes

- `[ ]` MCQ：每个 item 多次随机选项顺序；答案位置严格均衡。
- `[ ]` Short answer：接受 canonical name + aliases；记录 exact/normalized/semantic 三种评分。
- `[ ]` Image-text matching：判断图像是否对应目标概念，避免生成格式问题。

### P2 Indirect probes

- `[ ]` attribute、superclass、function/affordance、scene relation、description-to-image/image-to-description。
- `[ ]` 为每种 axis 定义最小可用 probe，不强迫所有数据集使用不自然的问题。
- `[ ]` 检查 indirect probe 是否泄漏 concept name；建立自动 lexical leakage checker。

### P3 Adversarial recovery

- `[ ]` prompt paraphrase ensemble。
- `[ ]` concept aliases、拼写变体和上/下位描述。
- `[ ]` multi-turn recovery 与 conflicting instruction。
- `[ ]` 图像 crop、resize、blur、color shift；只使用不改变 GT 的变换。
- `[ ]` 报告固定 attack budget 下的 worst-case，而不是无限重试后的最好结果。

**对应现有文件**：扩展 `vqa_gen/templates/`；重构 `vqa_gen/pipeline/build.py`；加强 `vqa_gen/pipeline/qc.py`。

**验收标准**：每个主概念至少有 direct + 一种 indirect probe；每个主表结果同时报告正常图像和 grounding controls；MCQ 与 free-form 结论不冲突或有解释。

## B-P0.5 Model-specific concept eligibility

- `[ ]` 在任何 unlearning 前，对每个 `model × concept × probe` 跑 `M0` baseline。
- `[ ]` 预注册 eligibility 规则：
  - `Acc_image ≥ τ_acc`；
  - `Acc_image − max(Acc_no_image, Acc_option_only, Acc_shuffled) ≥ τ_visual`；
  - bootstrap lower bound 或最小样本量达到要求；
  - 多 prompt/order variants 稳定性达到要求。
- `[ ]` 生成 `eligible_concepts/<model_revision>.json`，记录通过/失败原因。
- `[ ]` 主指标只在 eligible concepts 上汇总；附录同时给所有候选概念，防止 cherry-picking。
- `[ ]` 报告每个模型的 coverage：通过概念数/候选概念数。不能只报 eligible 子集上的高分。
- `[ ]` 概念选择阈值在 full run 前冻结，并做阈值敏感性分析。

**验收标准**：任何“忘得很好”的样本都能证明模型在干预前知道它且依赖图像；不存在 baseline 接近 chance 却被计作成功遗忘的概念。

## B-P0.6 指标与统计重构

核心不使用单一总分，至少报告以下维度：

- `[ ]` **Knowledge Access**：每个 probe 的原始正确率/匹配分数。
- `[ ]` **Worst-Case Leakage (WCL)**：固定攻击预算内，跨 direct/indirect/adversarial probes 的最大可恢复访问率。
- `[ ]` **Forgetting Effect**：相对 `M0` 的 paired access reduction；明确 chance floor 和 invalid/refusal 处理。
- `[ ]` **Matched Retain Fidelity**：干预前后 matched retain item 的正确性与输出一致性。
- `[ ]` **Neighborhood Damage**：相邻类、同 superclass、同 attribute 与远距离 retain 分层报告。
- `[ ]` **General Utility Retention**：至少覆盖知识、推理、感知三个维度，而不是只用单一 MMMU。
- `[ ]` **Refusal/Invalid Rate**：单独报告，不能把无效输出直接当作忘记成功。
- `[ ]` **Cost**：训练 GPU-hours、更新参数比例、额外推理延迟、峰值显存、API cost。
- `[ ]` **Coverage**：eligible concepts 比例和 method–model capability coverage。
- `[ ]` paired bootstrap 95% CI；关键二分类变化用 paired test；多概念/多模型汇总避免把样本当独立模型重复计数。
- `[ ]` 保存 per-item prediction，不只保存 aggregate；统计脚本从冻结记录自动生成 paper tables。

**对应现有文件**：拆分 `compute_metrics()`；替换 `scripts/aggregate_results.py` 的固定 condition 列表。

**验收标准**：拒答、格式失败、随机猜测、知识泄漏和 retain damage 在指标上可被区分；主结论带不确定性范围。

## B-P0.7 最小方法矩阵

### R0：Black-box I/O / behavioral suppression

- `[ ]` No-op / untreated baseline。
- `[ ]` 现有 prompt suppression，但做 held-out prompt tuning：开发集选 prompt，测试集冻结；报告 best single 与 prompt ensemble。
- `[ ]` In-context refusal/example baseline 或 self-critique baseline，二选一作为更强 black-box baseline。
- `[ ]` Semantic output filter：作为“最容易把输出藏起来”的下界/对照，明确不称为 unlearning。

### R1：Inference-state intervention

- `[ ]` 至少接入一种 representation/activation steering 方法，例如 R-MUSE 类方案。
- `[ ]` 至少接入一种 decoding/logit-level suppression 对照；只在输出空间与概念 token 映射有效的 probe 上使用，并报告适用范围。

### R2：White-box weight intervention

- `[ ]` Gradient Ascent 作为简单下界。
- `[ ]` NPO 或等价强训练式 baseline。
- `[ ]` 至少一种 multimodal-specific 方法：优先从 MANU、MMUnlearner、SIU 中选兼容性最高者。
- `[ ]` 对所有训练式方法记录是否用 retain set、训练步数、学习率搜索空间、更新参数比例和 checkpoint selection rule。

公平性要求：

- 方法在相同 forget/retain train split 上调参；最终 test 只运行一次冻结配置。
- 每个方法有相同或清晰披露的 tuning budget。
- 先 regime 内比较；跨 regime 比较必须同时给权限和成本。
- 不能把仅适配某一模型的方法失败泛化为该类方法整体失败。

**验收标准**：主 depth study 至少覆盖三个 access regimes，且每个 regime 有 no-op 之外的可运行 baseline；prompt 不再是唯一 intervention。

## B-P0.8 Acquisition-confound 审计实验

这是验证论文动机的关键实验，不是额外 appendix。

- `[ ]` 选 1–2 个代表性 synthetic/fictitious multimodal unlearning 设置。
- `[ ]` 对同一模型保存三个 checkpoint：`M0`、`M_acq`、`M_u`。
- `[ ]` 在完全相同的 general utility、retain 和 benchmark probes 上评测三者。
- `[ ]` 分解：
  - `Δ_acq = U(M_acq) − U(M0)`
  - `Δ_unlearn = U(M_u) − U(M_acq)`
  - `Δ_total = U(M_u) − U(M0)`
- `[ ]` 至少加入一个 matched-compute control：使用同样训练步数/数据量做非 forget-specific post-training，检查下降是否只是训练本身造成。
- `[ ]` 对超参数和随机种子做最低限度复现，避免用单个失败 fine-tuning run 支撑宏大主张。

**验收标准**：论文可以用数据回答“总能力下降有多少发生在 unlearning 之前”；如果 acquisition damage 很小，应诚实改写动机为“潜在且未被现有协议分解的 confound”，而不是强称结构性缺陷。

## B-P0.9 两层实验设计，控制组合爆炸

### Depth study：方法比较

- 3 个有代表性的 open-weight VLM：覆盖不同 family/size，且支持 R1/R2。
- 通过 eligibility 的核心 concept 子集。
- 全部 P0–P3 probes。
- 完整 R0/R1/R2 方法矩阵。
- 至少 3 个随机种子用于训练式方法；确定性方法报告 prompt/order/bootstrap 变化。

### Breadth study：模型生态比较

- 目标 10–13 个模型，覆盖 instruct/thinking、dense/MoE、small/medium/large。
- 只跑 no-op + 1–2 个 R0 方法 + 核心 probes，避免对每个模型跑全部白盒方法。
- 至少 2 个 closed frontier VLM；如果预算/条款不允许，标题、摘要和结论必须明确限定为 open-weight VLMs。
- 对 API 模型记录精确 model version、调用日期和参数。

### General utility suite

- `[ ]` 视觉知识/推理：MMMU。
- `[ ]` 视觉依赖与 leakage 控制：MMStar 或同类。
- `[ ]` 基础视觉感知/识别：选择一个稳定、成本可控的 benchmark。
- `[ ]` 可选：OCR、图表/文档或 hallucination，作为 P1 扩展。

**验收标准**：实验矩阵先生成 dry-run manifest 和预算表；主表不因缺失组合而产生不公平平均。

## B-P0.10 Construct-validity pilot

- `[ ]` 选择 2 个模型、3 个 concept axes、每 axis 3–5 个概念。
- `[ ]` 每个概念至少 50 个正常图像样本，并生成 no-image、shuffled、option-only 对照。
- `[ ]` 跑 MCQ、short answer、一个 indirect probe 和 prompt paraphrases。
- `[ ]` 人工检查不少于 200 个生成 item：图像正确性、GT、distractor、alias、问题自然度。
- `[ ]` 对 free-form scorer 做双人/模型-人工一致性抽查；目标一致率 ≥ 90%，否则不能进入 full run。
- `[ ]` 检查 answer position、question template、dataset source 对结果的影响。
- `[ ]` 输出一份 pilot report：保留/删除哪些 axes，为什么；冻结哪些阈值。

**验收标准**：通过 G0；所有被发现的 shortcut 有代码级防护或在 protocol 中显式报告。

## B-P0.11 Reproducibility、测试和 release hygiene

- `[ ]` 新增 `tests/test_schema.py`、`test_splits.py`、`test_prompt_leakage.py`、`test_metrics.py`、`test_model_contract.py`。
- `[ ]` 对 answer-order balance、split overlap、concept leakage、eligible filtering 写 deterministic tests。
- `[ ]` 添加 CPU-only tiny fixture 和 10-item smoke test；GPU full test 不放常规 CI。
- `[ ]` 固定 environment；不要只依赖精确 pin 的 `requirements.txt`，同时保存 CUDA/driver/transformers compatibility manifest。
- `[ ]` 每次结果运行写 `run_manifest.json`、`results.jsonl/parquet`、`metrics.json`、日志和失败记录。
- `[ ]` 数据 release 提供 checksum、license、download script、不可再分发数据的索引构建方式。
- `[ ]` README 改成 method-agnostic quickstart；旧 “In-Text Unlearning Evaluation” 移到 legacy 说明。

**验收标准**：从空环境按 README 能完成 tiny end-to-end；同一 seed 的数据/split hash 一致；paper table 可一条命令再生成。

## B-P1 强化实验

- `[ ]` **B-P1.1 Closed-model coverage**：至少两家独立提供商，记录版本漂移与费用。
- `[ ]` **B-P1.2 Scale/MoE analysis**：相同 family 内比较规模，避免把 family 差异误写成 scale law。
- `[ ]` **B-P1.3 Thinking vs instruct**：分别分析最终答案、可见 reasoning 中的概念泄漏和 refusal；不要披露不可公开 CoT。
- `[ ]` **B-P1.4 Semantic distractor audit**：用同 superclass/hard negative 与随机 negative 分层报告。
- `[ ]` **B-P1.5 Full-corpus statistics**：报告每 axis 的 candidate/eligible concepts、样本数、CI 和 power。
- `[ ]` **B-P1.6 Neighborhood damage**：目标类、同 superclass、近邻 attribute、远邻 retain 四层。
- `[ ]` **B-P1.7 Prompt robustness**：冻结 prompt bank，报告 mean、worst-case 和方差。
- `[ ]` **B-P1.8 Cross-format consistency**：MCQ、short answer、matching 的结论一致性矩阵。
- `[ ]` **B-P1.9 Attack adaptivity**：区分 oblivious attacker 与知道 method 的 adaptive attacker。
- `[ ]` **B-P1.10 Efficiency frontier**：忘记效果、utility、latency/GPU-hours 三维比较。

## B-P2 后续扩展

- `[ ]` 更广隐私类别：非名人、地点、个人物品或可撤回授权的实体，但需先解决许可与伦理。
- `[ ]` 组合概念和关系 unlearning，而非仅单标签概念。
- `[ ]` 持续集成新方法的 leaderboard；提交必须包含 capability 和 compute disclosure。
- `[ ]` 提供 Hugging Face dataset release、versioned benchmark server 和 result submission validator。
- `[ ]` 扩展到视频 VLM 或多图对话，不能影响 AAAI 主线进度。

---

## 4. 代码任务与论文 claim 的绑定

| 论文 claim | 必需代码/实验 | 主产物 | 通过标准 |
|---|---|---|---|
| 评测 pre-existing knowledge | eligibility pipeline + `M0` manifest | Figure 2 / dataset table | 主结果只含 baseline-known 且 visually grounded concepts |
| benchmark method-agnostic | method/model/probe 独立接口 | capability matrix | R0/R1/R2 都有可运行方法，runner 不含方法特判 |
| 不把输出抑制当遗忘 | free-form、indirect、adversarial、refusal metrics | leakage heatmap | direct 下降与 worst-case leakage 分别可观测 |
| 避免 learn-then-forget 混淆 | `M0/M_acq/M_u` 三状态 audit | Table 3 | 分开报告 `Δ_acq` 与 `Δ_unlearn` |
| 保留能力评测可信 | matched retain + general utility suite | Pareto figure | 目标、近邻、远邻和 general utility 分层 |
| 结果可复现 | revisioned configs + run manifest + tests | artifact appendix | 表格可从冻结结果自动生成 |

---

## 5. Reviewer 问题到修改项的映射

| Review concern | 必须回应的位置 | Roadmap ID |
|---|---|---|
| Prompt suppression 不等于 unlearning | Title、task definition、method taxonomy、limitations | A-P0.1, A-P0.5, A-P0.8 |
| Benchmark prescribed one prompt/method | method adapter + 三 access regimes | B-P0.2, B-P0.7 |
| 没有 best-effort tuned baseline | held-out tuning budget、prompt ensemble、强 baseline | B-P0.7 |
| Fine-tuning bottleneck/能力下降证据不足 | 三状态 acquisition audit | B-P0.8 |
| Oracle anomaly/含义不清 | 降级为 instruction-following control，不进主方法排名 | A-P0.5, B-P0.2 |
| MCQ 与 distractor confound | short answer、matching、order balance、semantic negatives | B-P0.4, B-P1.4 |
| 不清楚模型是否看图 | no-image、shuffled、option-only eligibility | B-P0.4, B-P0.5 |
| Prompt robustness 不足 | held-out prompt bank、paraphrase worst-case | B-P0.4, B-P1.7 |
| 无 closed/large/MoE | breadth study 或缩窄 claim | B-P0.9, B-P1.1, B-P1.2 |
| General utility 太窄 | MMMU + MMStar + perception suite | B-P0.6, B-P0.9 |
| Baseline 类型单一 | output filter、self-critique/ICL、activation、GA/NPO、multimodal method | B-P0.7 |
| Thinking model 分析不足 | reasoning-visible leakage 与最终答案分开 | B-P1.3 |
| Privacy breadth/伦理 | license audit、limitations、P2 expansion | A-P0.8, B-P0.11, B-P2 |

---

## 6. 建议里程碑（按依赖，不绑定具体日期）

### M0：定义冻结

- 完成 A-P0.1、A-P0.5。
- 确认题目、claim 边界、access regimes、probe taxonomy、eligibility 初始阈值。
- 产物：`benchmark_spec_v1.md`、schema 草案、Figure 1 草图。

### M1：核心重构

- 完成 B-P0.1–B-P0.6 的最小实现。
- 把现有 in-text 方法迁成插件；旧脚本保留兼容 wrapper。
- 产物：可运行的 no-op + prompt baseline + P0/P1 probes + 新 metrics。

### M2：Pilot 与 Go/No-Go

- 完成 B-P0.10。
- 冻结 concept axes、thresholds、prompt bank、scorer 和 full-run 样本量。
- 未通过 G0 时不得启动大规模 sweep。

### M3：方法与 confound audit

- 完成 B-P0.7、B-P0.8。
- 先在一个模型跑通所有方法，再扩到 depth study 三个模型。
- 产物：method capability matrix、三状态 audit 初表。

### M4：Full experiments

- 先 depth study，确认指标和 checkpoint 无误后再 breadth study。
- 同步完成 general utility 和 cost logging。
- 结果冻结后打 tag，生成 paper tables/figures。

### M5：论文成稿

- 按 A-P0.2–A-P0.8 写完整初稿。
- 做一次“claim–evidence audit”：每个强 claim 必须指向表/图/统计结果。
- 做一次“reviewer simulation”：让未参与项目的人只看 Abstract、Figure 1、Introduction 和主表，复述论文贡献。

### M6：AAAI submission readiness

- 匿名化代码/链接与 supplement。
- 检查 AAAI 最新格式、页数、伦理与 reproducibility checklist。
- 冻结 paper PDF、appendix、code commit、data version 和 result manifests。

---

## 7. 最小可投稿标准

以下项目全部完成，才进入投稿冻结：

- `[ ]` Abstract/Introduction 不再把 prompt suppression 等同于 unlearning。
- `[ ]` Figure 1 清楚解释 learn-then-forget confound 和 ULBench solution。
- `[ ]` 候选概念经过 model-specific knowledge + visual grounding eligibility。
- `[ ]` 主评测同时包含 MCQ 与至少一种非 MCQ 格式。
- `[ ]` 有 direct、indirect、adversarial probes 和 fixed attack budget。
- `[ ]` benchmark runner 与 methods 解耦。
- `[ ]` R0/R1/R2 均有至少一个实际 baseline，而不是只写接口。
- `[ ]` 有 `M0/M_acq/M_u` acquisition-confound audit。
- `[ ]` 有 matched retain、general utility、refusal/invalid、cost 和 CI。
- `[ ]` depth 与 breadth study 分开，避免不完整笛卡尔积造成误导。
- `[ ]` closed model 缺失时已明确缩窄 claim；若保留 broad VLM claim，则至少加入 2 个 closed models。
- `[ ]` 数据 license/provenance 和 celebrity/logo ethics 经过审查。
- `[ ]` 所有主表可由冻结 manifest 一键生成。
- `[ ]` README、配置、模型清单与实际可运行代码一致。
- `[ ]` 至少一位未参与开发的人复现 tiny end-to-end。

---

## 8. 需要尽快做出的决策

| 决策 | 推荐默认值 | 截止点 | 状态 |
|---|---|---|---|
| 最终标题 | “Beyond Learn-Then-Forget…” | M0 | `[ ]` |
| 核心概念 axes | 先由 pilot eligibility 决定，不强保全部 11 axes | M2 | `[ ]` |
| `τ_acc` / `τ_visual` | pilot 前预注册候选值，pilot 后只做有记录的冻结 | M2 | `[ ]` |
| Depth study 模型 | 3 个不同 family、支持白盒的 open VLM | M2 | `[ ]` |
| Closed models | 至少 2 个；否则缩窄 claim | M3 前 | `[ ]` |
| R1 方法 | 优先能稳定复现且支持目标模型的 representation steering | M3 前 | `[ ]` |
| R2 multimodal 方法 | MANU/MMUnlearner/SIU 中选兼容性最高者 | M3 前 | `[ ]` |
| General utility suite | MMMU + MMStar + 1 个 perception benchmark | M2 | `[ ]` |
| Open-answer scorer | normalization + alias + semantic judge + 人工抽查 | M2 | `[ ]` |
| 是否保留 ImageNet | 补齐 config 和 release 流程，否则从当前支持清单删除 | M1 | `[ ]` |

---

## 9. 下一步：最先执行的 10 个任务

1. `[x]` 写 `benchmark_spec_v1.md`：任务、权限、probe、metrics、claim boundary。
2. `[x]` 修正 `split_registry.yaml` 命名/`k` 不一致及 README/ImageNet 不一致。
3. `[x]` 新建 `ulbench/schema.py` 与旧 JSONL validator/migration。
4. `[ ]` 抽离 `ModelAdapter` 和 `UnlearningMethod`；迁移现有 prompt conditions。
5. `[ ]` 实现 no-image、shuffled-image、option-only controls。
6. `[ ]` 实现 MCQ option randomization + short-answer probe + alias scorer。
7. `[ ]` 实现 eligibility manifest 与 coverage reporting。
8. `[ ]` 拆出新 metrics：WCL、matched retain、refusal、cost、bootstrap CI。
9. `[ ]` 在 2 个模型上跑 construct-validity pilot，并召开一次 G0 决策。
10. `[ ]` Pilot 通过后再接 R1/R2 方法和启动 full study；同时按新故事改 Abstract/Introduction。

---

## 10. 进展记录

| 日期 | 变更 | 证据/产物 | 对 claim 的影响 |
|---|---|---|---|
| 2026-07-20 | 基于拒稿意见和 `main@47a6376` 建立双线路重构计划 | 本文件 | 从 training-free/prompt benchmark 改为 method-agnostic pretrained visual concept unlearning benchmark |
| 2026-07-20 | 冻结 benchmark specification v1 normative draft | `docs/benchmark_spec_v1.md` | 明确 benchmark/method/evaluation 边界，并冻结 eligibility、probe、result status 与 metric contracts |
| 2026-07-20 | 修复 split registry 和 ImageNet 文档/配置漂移 | `scripts/split_registry.yaml`, `vqa_gen/configs/imagenet_identity_mvp.yaml`, `tests/test_repository_consistency.py` | 防止 split 规模误报，并将 ImageNet 支持限定为可验证的显式配置 |
| 2026-07-20 | 新建 v1 schema、跨记录 validator 与 legacy migration CLI | `ulbench/schema.py`, `ulbench/validation.py`, `ulbench/tools/`, `tests/` | 将 eligibility、method capability、manifest 和 invalid/refusal 边界变成可执行合同；legacy 数据不能静默进入 v1 |
