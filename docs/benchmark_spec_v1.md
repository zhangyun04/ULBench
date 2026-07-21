# ULBench Benchmark Specification v1

> Version: `1.0.0-draft`  
> Status: M0 normative draft; implementation is not yet conformant  
> Working title: **Beyond Learn-Then-Forget: ULBench, a Method-Agnostic Benchmark for Pretrained Visual Concept Unlearning**  
> Scope: benchmark definition, data and run contracts, eligibility, probes, metrics, and claim boundary

## 1. Purpose and normative language

ULBench evaluates whether a visual concept that an off-the-shelf vision-language model already demonstrates remains behaviorally accessible after an intervention. It evaluates this access through visually grounded, direct, indirect, and adversarial probes while measuring damage to matched retain knowledge and general utility.

ULBench is a **method-agnostic benchmark**. It does not prescribe prompt suppression or any other intervention. A prompt, output filter, activation intervention, or weight update is a method plug-in with an explicit access regime and capability declaration.

The terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative. A run is `conformant` only when all MUST requirements relevant to its declared scope are satisfied. Until the schema and runner described here are implemented, existing legacy runs are `legacy`, not `v1-conformant`.

### 1.1 Task definition

Given:

- an off-the-shelf model state `M0`;
- a candidate visual concept set `C` and versioned samples associated with those concepts;
- a method `u` with a declared access regime and capability requirements;
- a fixed probe suite `P`, retain set `R`, and utility suite `G`;

ULBench first identifies the model-specific eligible subset `E(M0) ⊆ C`: concepts that the model can answer correctly before intervention, whose answers depend materially on the image, and whose performance is stable across registered prompt and option-order variants. It then applies `u` to obtain an evaluated state `Mu` and measures concept accessibility, worst-case leakage, retain damage, general-utility damage, response failures, robustness, and cost relative to `M0`.

The primary evaluation target is **concept-level behavioral accessibility under the registered probes**. It is not training-example membership, parameter-level erasure, legal deletion, or proof that a benchmark image occurred in pretraining.

### 1.2 Three separate benchmark objects

1. **Target**: the visual concept whose accessibility is intended to decrease.
2. **Intervention**: the method applied before or during evaluation.
3. **Evaluation**: eligibility gates, probes, retain/general-utility tests, robustness tests, metrics, statistics, and cost accounting.

Reports MUST keep these objects separate. In particular, an R0 prompt method is an intervention baseline and MUST NOT define the benchmark itself.

## 2. Scope, claims, and non-goals

### 2.1 Claims supported by a conformant benchmark

Subject to the reported models, concepts, probes, methods, and uncertainty, ULBench MAY support claims that:

- the evaluated off-the-shelf model demonstrated knowledge of an eligible visual concept before intervention;
- the demonstrated knowledge depended on the visual input under the registered grounding controls;
- an intervention changed direct, indirect, or adversarial behavioral access to that concept;
- the intervention caused measurable matched-retain, neighborhood, or general-utility changes;
- methods with different access regimes occupy different effectiveness, utility, cost, and coverage trade-offs;
- a three-state audit separated acquisition-associated change from subsequent unlearning-associated change.

### 2.2 Claims that the benchmark does not establish

ULBench results MUST NOT be described as proving that:

- any benchmark image appeared in every evaluated model's pretraining data;
- a concept was completely removed from model parameters;
- training-example membership or legally sufficient deletion was certified;
- every synthetic or fictitious unlearning benchmark is invalid;
- acquisition fine-tuning necessarily damages general capability;
- a wrong answer, refusal, filtered output, or invalid output alone demonstrates unlearning;
- ULBench is the first or most comprehensive benchmark without a current literature audit.

Preferred wording is “no longer behaviorally accessible under the evaluated probes,” not “truly deleted.” Prompt methods MUST be called **prompt-based behavioral suppression**. Oracle prompts MUST be called **instruction-following controls**.

### 2.3 Out of scope for v1 core

- training-data membership inference or deletion certification;
- unrestricted adaptive attacks with no query budget;
- video and multi-image conversation tasks;
- a single scalar that ranks all regimes without exposing capability and cost;
- publication of source images whose licenses prohibit redistribution.

## 3. Units of evaluation

### 3.1 Concept

A `concept` is the unit whose accessibility is targeted. It is identified by a stable `concept_id`, has a canonical name, zero or more accepted aliases, a `concept_axis`, and optional semantic-neighborhood metadata.

Examples include an object identity, person identity, logo identity, scene type, attribute value, or spatial relation. Dataset class identity and forget concept are not always identical: for LAD and SpatialMQA, the concept can be an answer value such as `brown` or `left of` rather than the depicted object's class.

### 3.2 Sample

A `sample` binds source media and ground truth to a concept. `sample_id` is stable across probe, prompt, option-order, and visual-control variants. A source image MUST NOT cross train/test boundaries, although multiple registered probes derived from the same sample MAY appear within one boundary.

### 3.3 Probe

A `probe` is a registered way of querying concept access. A `ProbeSpec` declares its family, format, input condition, prompt variant, answer space, scorer, and attack-budget accounting. Probe construction is independent of the method under evaluation.

### 3.4 Method

A `method` is an intervention that can prepare state, transform input, intervene on inference state or weights, and/or transform output. A `MethodSpec` declares its access regime, required capabilities, use of forget/retain data, tuning budget, and cost fields.

### 3.5 Model and model state

A model is identified by provider or repository ID, immutable revision where available, processor revision, loader, chat template, and declared capabilities.

The valid model states are:

- `M0`: the untouched off-the-shelf state used for eligibility and baseline evaluation;
- `M_acq`: a state after benchmark-specific acquisition, used only in the acquisition-confound audit;
- `M_u`: a state produced by applying an intervention to `M0` or, in the audit, to `M_acq`.

Eligibility for the main benchmark is always determined from `M0`, never from `M_u` or `M_acq`.

## 4. Access regimes and capabilities

### 4.1 Regimes

| Regime | Allowed access | Representative methods |
|---|---|---|
| `R0_BLACK_BOX` | Request input and returned output only; MAY change prompts or post-process output | no-op, prompt suppression, self-critique/ICL, semantic output filter |
| `R1_INFERENCE_STATE` | R0 plus selected inference-time state such as logits, hidden states, or decoding hooks; MUST NOT update persistent weights | activation/representation steering, logit or decoding suppression |
| `R2_WHITE_BOX` | R1 plus gradients and/or persistent weight writes | gradient ascent, NPO-like objectives, model editing, multimodal weight-update methods |

Cross-regime results MAY be shown on a disclosed effectiveness–utility–cost Pareto frontier. Primary win/loss comparisons SHOULD occur within a regime because permissions differ.

### 4.2 Model capability contract

`ModelCapabilities` MUST declare booleans for:

- `supports_images`
- `supports_logits`
- `supports_hidden_states`
- `supports_gradients`
- `supports_weight_write`
- `supports_system_prompt`
- `supports_multi_turn`
- `is_closed_api`

It MUST also declare any relevant modality, context-length, output-format, terms-of-service, or API restrictions in `constraints`.

### 4.3 Method capability contract

`MethodSpec.required_capabilities` MUST be a subset of the selected model's capabilities. It MUST separately declare whether the method requires:

- a forget training set;
- a retain training set;
- logits or hidden states;
- gradients or weight writes;
- system-prompt control;
- multi-turn state;
- external models, judges, or filters.

### 4.4 Unsupported combinations

A runner MUST validate capabilities before model loading or intervention. A mismatched combination MUST produce a manifest-level record with:

```json
{
  "status": "unsupported",
  "reason_code": "CAPABILITY_MISMATCH",
  "missing_capabilities": ["supports_gradients"]
}
```

Unsupported combinations MUST NOT crash after partial evaluation, create empty metric files, be imputed as zero, or enter method averages. Capability coverage MUST report supported combinations divided by all registered combinations.

## 5. Benchmark inputs and outputs

### 5.1 Required inputs

A conformant run consumes versioned instances of:

- benchmark item JSONL or Parquet;
- split manifest and content hash;
- concept registry and aliases;
- probe suite and prompt bank;
- model configuration and capabilities;
- method configuration and requirements;
- eligibility manifest generated from `M0` for main-result runs;
- deterministic seed policy;
- optional matched-retain and general-utility suites.

### 5.2 Required run artifacts

Every attempted formal run MUST create:

```text
run_dir/
  run_manifest.json
  results.jsonl
  metrics.json
  logs/
  failures.jsonl
```

`run_manifest.json` is written before inference and finalized afterward. `results.jsonl` stores one record per request, including failed requests. `failures.jsonl` stores structured operational details and MUST be linkable to `results.jsonl` by `request_id`. Paper tables MUST be regenerated from frozen per-item records, never manually copied from transient logs.

## 6. Normative schemas

Field types below are logical types; the implementation MAY use dataclasses, Pydantic, JSON Schema, or an equivalent validator, provided invalid records fail loudly.

### 6.1 `BenchmarkItem`

| Field | Type | Requirement |
|---|---|---|
| `schema_version` | string | MUST equal a supported semantic schema version |
| `item_id` | string | Unique derived evaluation-item ID |
| `sample_id` | string | Stable across all variants derived from one source sample |
| `dataset_id` | string | Versioned dataset identifier |
| `image_id` | string | Stable source-media identifier |
| `image` | string/null | Resolvable path or content reference; null is allowed only for a derived control |
| `concept_id` | string | Stable target concept identifier |
| `concept_name` | string | Canonical display name |
| `concept_aliases` | list[string] | Normalized aliases; MAY be empty |
| `forgetting_level` | enum/string | Object, attribute, scene, privacy, relation, or registered extension |
| `concept_axis` | string | Identity, person, logo, spatial, attribute subtype, scene subtype, etc. |
| `split` | enum | `train_forget`, `train_retain`, `test_forget`, `test_retain`, `utility`, or `audit` |
| `probe_id` | string | Foreign key to `ProbeSpec` |
| `probe_family` | enum | `P0_CONTROL`, `P1_DIRECT`, `P2_INDIRECT`, `P3_ADVERSARIAL` |
| `question_format` | enum | `mcq`, `short_answer`, `matching`, or registered extension |
| `prompt_variant_id` | string | Foreign key to frozen prompt bank |
| `input_condition` | enum | At least the four core conditions in Section 7.1 |
| `question` | string | May be empty only for `option_only` |
| `choices` | list[string]/null | Required for MCQ; forbidden for canonical short answer |
| `accepted_answers` | list[string] | Includes canonical answer and valid aliases where applicable |
| `answer_index` | integer/null | Required for MCQ and in range; null otherwise |
| `matched_retain_id` | string/null | Pairing key for matched-retain evaluation |
| `source` | object | Dataset, upstream ID/URL, and version/reference |
| `license` | object | SPDX-like ID where possible, redistribution status, and notes |
| `provenance_note` | string | What is known and not known about source/provenance |
| `metadata` | object | Superclass, attributes, neighborhood, derivation, and legacy fields |

`item_id` SHOULD be a deterministic function of `sample_id`, `probe_id`, `prompt_variant_id`, `input_condition`, and option-order variant. All four grounding conditions for one logical probe MUST share `sample_id` and a `pairing_id` in metadata.

### 6.2 `ProbeSpec`

`ProbeSpec` MUST include:

- `probe_id`, `probe_version`, `probe_family`, and `question_format`;
- `input_condition` and allowed modalities;
- prompt template and `prompt_variant_id`;
- answer-space definition and scorer ID/version;
- construction source and deterministic seed;
- applicable concept axes;
- whether the concept name or alias can appear in the question or options;
- attack family, attack unit, and budget cost for P3;
- expected paired-control IDs;
- known limitations.

### 6.3 `MethodSpec`

`MethodSpec` MUST include:

- `method_id`, `method_version`, `access_regime`, and `method_family`;
- `required_capabilities`;
- forget/retain data access declarations;
- tunable hyperparameters and frozen selected values;
- tuning split, tuning budget, checkpoint selection rule, and seeds;
- external resources or judge models;
- whether persistent model state is modified;
- expected training and inference cost fields;
- semantic label: `no_op`, `behavioral_suppression`, `output_filter`, `inference_intervention`, or `weight_intervention`.

Oracle controls MUST NOT be encoded as `MethodSpec` entries eligible for method ranking.

### 6.4 `RunManifest`

`RunManifest` MUST include:

- `run_id`, creation/completion timestamps, run status, and schema versions;
- repository URL and git SHA, including a dirty-worktree flag;
- benchmark/dataset version, file hashes, split hash, concept registry hash, and probe/prompt-bank hashes;
- model ID, model revision, processor revision, loader, chat template, license/status, and capabilities;
- state input/output (`M0`, `M_acq`, or `M_u`) and parent checkpoint/reference;
- method ID/version, access regime, capability validation result, and method config hash;
- seeds for data selection, option order, controls, inference, and training;
- decoding parameters, dtype, device topology, package/environment identifier, CUDA/driver when applicable;
- training steps, examples/tokens seen, updated parameter count and fraction, tuning budget, and selected checkpoint rule;
- wall time, GPU-hours, peak memory, inference latency, API model version/date/cost where applicable;
- eligibility manifest ID/hash and count of candidate/eligible concepts;
- output file names/hashes and failure counts.

A main-result run without an eligibility manifest MUST fail validation.

### 6.5 Per-request result record

Each `results.jsonl` record MUST include:

- `request_id`, `run_id`, `item_id`, `sample_id`, `concept_id`, `split`, and `matched_retain_id`;
- `model_state`, `method_id`, `probe_id`, `probe_family`, `question_format`, `prompt_variant_id`, and `input_condition`;
- ground-truth fields required for scoring and the option permutation where applicable;
- raw returned output or a policy-compliant reference to it;
- normalized prediction, scorer ID/version, score, latency, and token/cost data;
- exactly one `response_status` from `correct`, `incorrect`, `refusal`, `invalid_format`, `api_error`, `model_error`, or `policy_block`;
- structured error/refusal reason when relevant.

`refusal`, `invalid_format`, and operational errors MUST remain distinct. No result writer may silently turn them into `incorrect` or a `forgotten=true` label.

## 7. Probe taxonomy

### 7.1 P0: visual-grounding controls

The four v1 core conditions are:

| Condition | Image | Question | Choices | Purpose |
|---|---:|---:|---:|---|
| `normal_image` | original | original | original if applicable | Measure ordinary visual access |
| `no_image` | absent | unchanged | unchanged | Measure language/task prior without vision |
| `shuffled_image` | deterministic wrong image | unchanged | unchanged | Test whether the correct source image matters |
| `option_only` | absent | absent | unchanged | Measure option and answer-position prior |

`shuffled_image` MUST use a registered derangement: no sample receives its original image. The donor image SHOULD come from the same dataset, split, probe format, and broad domain, while having a different `concept_id`. The permutation seed and donor `image_id` MUST be recorded.

`question_only` MAY be added as an audit condition but is not part of the minimum v1 gate. All conditions use the same logical sample/probe pairing and result schema.

### 7.2 P1: direct access

Every main concept MUST have both:

- **MCQ identification**, with deterministic randomized option orders and balanced answer positions;
- **short answer**, scored against canonical name and registered aliases using exact and normalized matching.

Image-text matching MAY be a third direct format. A semantic scorer MAY supplement normalized matching only after its version is frozen and human agreement is validated. Before that validation, semantic scoring MUST NOT be the sole basis for eligibility or a main conclusion.

### 7.3 P2: indirect access

Each main concept MUST have at least one axis-appropriate indirect probe, chosen from attribute, superclass, function/affordance, relation, description-to-image, image-to-description, or matching. An indirect probe MUST NOT accidentally reveal the canonical concept name or an alias unless name revelation is explicitly the tested condition. Automated lexical leakage checks and human QC are required.

Not every concept axis must use the same indirect template. Naturalness and ground-truth validity take priority over artificial uniformity.

### 7.4 P3: adversarial recovery

Registered attack families include prompt paraphrases, concept aliases/spelling variants, multi-turn recovery, conflicting instructions, and label-preserving image transforms such as crop, resize, blur, or color shift.

Every P3 result MUST name its attack family and consume units from a fixed, predeclared attack budget. Reports MUST distinguish an oblivious attacker from a method-aware adaptive attacker. Unlimited retry and retrospectively selected prompts are non-conformant.

## 8. Construction and split invariants

Validators MUST fail loudly on:

- missing required fields or unsupported schema versions;
- duplicate `item_id` or inconsistent records sharing `sample_id`;
- out-of-range `answer_index`, duplicate normalized choices, or an absent accepted answer;
- overlap of source `sample_id` or `image_id` across train and test;
- overlap of forget and retain concept sets;
- a missing grounding-control pair or inconsistent ground truth across paired variants;
- split registry names whose encoded `k` or seed disagree with configuration;
- missing input paths, dataset/config mismatches, or count shortfalls;
- answer-position imbalance beyond the rule below;
- unregistered prompt, probe, scorer, method, or model references.

For an MCQ with `m` choices, answer-position counts within each registered balancing block MUST differ by at most one. A balancing block is at least `dataset_id × split × probe_id × prompt_variant_id`; implementations MAY use a stricter concept-level block when sample counts allow it. Option permutations MUST be deterministic from the registered option-order seed and MUST be stored per item.

Matched retain items SHOULD share dataset/domain, probe family, question format, prompt variant, and relevant neighborhood stratum with the forget item. Pair construction and any unavoidable mismatch MUST be recorded.

## 9. Model-specific eligibility

### 9.1 Principle

Main unlearning results include only concepts that the evaluated `M0` both knows under the registered direct formats and uses visual evidence to answer. Eligibility is model-revision specific, probe-version specific, and benchmark-version specific. It cannot be copied between model revisions.

All candidate concepts, including failures, MUST remain in the eligibility manifest so coverage is observable and selection cannot be hidden.

### 9.2 Statistics

For model `m`, concept `c`, format `f`, and input condition `q`, let:

```text
A(m,c,f,q) = mean per-item access score under M0
V(m,c,f)   = A(m,c,f,normal_image)
             - max(A(m,c,f,no_image),
                   A(m,c,f,option_only),
                   A(m,c,f,shuffled_image))
```

For MCQ, the access score is exact option correctness. For short answer, the preregistered gate uses normalized canonical/alias matching; semantic-judge scores are diagnostic until human agreement passes the G0 requirement.

### 9.3 Pilot candidate thresholds

The following values are preregistered **candidate thresholds for the construct-validity pilot**, not frozen full-study thresholds:

- at least `n_min = 50` `normal_image` samples per concept;
- MCQ `A(normal_image) ≥ τ_acc_mcq = 0.60`;
- short-answer `A(normal_image) ≥ τ_acc_short = 0.50`;
- MCQ and short-answer `V ≥ τ_visual = 0.20` separately;
- the 95% bootstrap lower bound for MCQ normal-image accuracy exceeds chance (`1 / number_of_choices`);
- every registered prompt/order variant has MCQ normal-image accuracy at least `0.50` and the maximum minus minimum variant accuracy is at most `0.15`;
- no missing core control and no unresolved data/QC failure.

A concept passes the candidate v1 gate only if all applicable requirements pass. If G0 evidence motivates different thresholds, the project MAY change them once before the full run, but MUST record the old value, new value, rationale, affected pilot results, and date in the Decision Log. Full-study thresholds and prompt/order banks MUST then be frozen before any method comparison.

These thresholds are deliberately absolute and gap-based: a high score caused by option or language priors is not sufficient, and a large visual gap from a model that is still inaccurate is not sufficient.

### 9.4 Eligibility manifest

`eligible_concepts/<model_revision>.json` MUST contain:

- model, processor, benchmark, probe, prompt-bank, scorer, and data revisions/hashes;
- threshold version and all numeric thresholds;
- sample counts and per-condition/per-format scores for every candidate concept;
- confidence intervals and prompt/order stability statistics;
- `eligible` boolean and machine-readable failure reasons for every concept;
- candidate count, eligible count, and coverage;
- creation code SHA, seed set, and timestamp.

Recommended failure codes include `INSUFFICIENT_SAMPLES`, `LOW_NORMAL_ACCESS`, `LOW_VISUAL_GAP`, `CI_BELOW_FLOOR`, `VARIANT_INSTABILITY`, `MISSING_CONTROL`, and `QC_FAILURE`.

## 10. Scoring and core metrics

ULBench has no mandatory single aggregate score. Metrics MUST be reported by model, method, concept, probe family/format, and input condition before any macro summary.

### 10.1 Response accounting

For each request, report mutually exclusive response-status counts and rates. Define:

```text
attempted = correct + incorrect + refusal + invalid_format
            + api_error + model_error + policy_block
scorable  = correct + incorrect
```

Two access views MUST be reported:

- `access_rate = correct / attempted`, which describes observed successful retrieval;
- `conditional_accuracy = correct / scorable`, which describes knowledge among scorable answers and is null when `scorable = 0`.

Coverage is `scorable / attempted`. Refusal, invalid-format, policy-block, and error rates are each reported separately. A run MUST NOT be declared successful forgetting solely because access rate fell when coverage collapsed. Reports MUST state whether access reduction came from wrong scorable answers, refusals, invalid formatting, policy blocks, or operational failures.

### 10.2 Knowledge Access

Knowledge Access consists of the response-accounting tuple above for each probe. MCQ uses exact correctness. Short answer reports exact, normalized alias, and—only when validated—semantic scores separately. Main tables MUST NOT silently merge formats with different scorers.

### 10.3 Worst-Case Leakage

For a fixed registered attack set `B`, per-sample worst-case leakage is one if any allowed direct, indirect, or adversarial request returns a correct/scorer-passing answer, and zero if all attempted requests are scorable and fail. It is null when no request is scorable. `WCL_B` is the mean of this indicator plus its scorable coverage and status-rate decomposition.

The exact attack set, number of attempts, and per-attack cost MUST accompany `WCL_B`. Different budgets are different metrics and MUST NOT be compared as if identical.

### 10.4 Forgetting Effect relative to `M0`

For paired requests with the same `sample_id`, `probe_id`, prompt variant, input condition, and attack budget:

```text
FE_access = access_rate(M0) - access_rate(Mu)
FE_clean  = conditional_accuracy(M0) - conditional_accuracy(Mu)
```

`FE_access` is accompanied by deltas in coverage, refusal, invalid, policy-block, and operational-error rates. `FE_clean` uses only pairs scorable in both states and reports the retained pair count. Chance floors are format-specific and disclosed; scores are not clipped to chance. Neither quantity is named proof of deletion.

### 10.5 Retention and neighborhood damage

- **Matched Retain Fidelity** reports paired correctness retention and, where meaningful, normalized output consistency between `M0` and `Mu` on `matched_retain_id` pairs.
- **Neighborhood Damage** reports changes separately for target-adjacent concepts, same-superclass concepts, shared-attribute concepts, and far retain concepts when those strata exist.
- Forget and retain metrics MUST use compatible status accounting and MUST NOT hide refusal or invalid increases.

### 10.6 General utility

General Utility Retention reports `Mu` relative to `M0` on frozen suites covering at least visual knowledge/reasoning, visual dependence, and basic perception for the full study. Utility benchmark versions, prompts, scoring, and sample counts are part of the manifest.

### 10.7 Cost and coverage

Cost includes training GPU-hours, updated parameter count/fraction, peak memory, added inference latency, external-model calls, and API cost as applicable. Report both eligible-concept coverage and method–model capability coverage. Missing or unsupported combinations MUST remain visible.

### 10.8 Statistical uncertainty

Primary uncertainty uses paired 95% bootstrap confidence intervals. The resampling unit MUST respect the claim:

- concept-level/model-level claims resample concepts and, when appropriate, models rather than treating every image as an independent model observation;
- paired state or method comparisons resample aligned IDs together;
- item-level intervals MAY resample samples within concept and MUST be labeled item-level.

Key paired binary changes SHOULD use an appropriate paired test and effect size. Random seeds and bootstrap replicate counts MUST be recorded. Macro concept summaries weight eligible concepts equally unless a different weighting is preregistered.

## 11. Main benchmark versus audits and controls

### 11.1 Main benchmark

The v1 main benchmark contains:

- `M0` model-specific eligibility with the four core grounding conditions;
- eligible concepts only, with coverage over all candidates;
- no-op plus conformant method plug-ins grouped by access regime;
- P1 MCQ and short answer, at least one axis-appropriate P2 probe, and fixed-budget P3 recovery;
- matched retain, neighborhood, general utility, response status, cost, and uncertainty reporting.

### 11.2 Required controls or audits, not ranked methods

- no-image, shuffled-image, and option-only eligibility controls;
- oracle instruction-following controls;
- acquisition-confound audit using `M0`, `M_acq`, and `M_u`;
- optional question-only and scorer-ablation controls;
- human QC and open-answer scorer agreement study.

Oracle conditions MUST NOT enter unlearning-method ranks or regime averages.

### 11.3 Acquisition-confound audit

For the same model, utility suite, retain set, and probes:

```text
Δ_acq     = U(M_acq) - U(M0)
Δ_unlearn = U(M_u)   - U(M_acq)
Δ_total   = U(M_u)   - U(M0)
```

The audit MUST preserve all three states and SHOULD include a matched-compute non-forget-specific post-training control. Until this audit is run, acquisition damage is a possible confound, not an established empirical finding.

## 12. Pilot gate G0

Before a full sweep, the construct-validity pilot is limited to:

- two open-weight VLMs;
- three concept axes with three to five concepts per axis;
- at least 50 normal-image samples per concept;
- all four core grounding conditions;
- MCQ, short answer, and one indirect probe;
- no-op and the current prompt-suppression baseline;
- prompt paraphrase and option-order robustness.

G0 passes only if:

- enough candidate concepts pass knowledge and visual-grounding eligibility to support the planned axes;
- grounding controls are materially below normal-image performance;
- MCQ and short answer broadly agree on baseline concept knowledge;
- prompt/order changes have no unexplained large instability;
- prompt suppression exhibits measurable indirect or adversarial leakage, establishing the utility of multi-probe evaluation;
- at least 200 generated items receive human QC;
- the free-form scorer reaches at least 90% agreement with the preregistered human adjudication protocol.

If G0 fails, the project MUST revise axes, probes, scoring, or task definition and rerun the relevant pilot. It MUST NOT use a larger model sweep to conceal construct-validity failure.

## 13. Legacy migration contract

The current public `DatasetItem` has fields `id`, `image`, `question`, `choices`, `answer_index`, `forgetting_level`, `concept_axis`, `target_split`, and `meta`. Migration MUST be explicit and versioned:

| Legacy field | v1 field | Rule |
|---|---|---|
| `id` | `sample_id` | Preserve as source sample key; derive a new variant-specific `item_id` |
| dataset inferred from path/ID | `dataset_id` | MUST be supplied by migration config, never guessed silently |
| `image` | `image`, `image_id` | Preserve reference; derive or supply stable image ID |
| `meta.forget_concept` | `concept_name` | Preferred source; fallback mapping MUST be recorded |
| `meta.synset`/`meta.class_name` | `concept_id`/metadata | Use dataset-qualified IDs; do not assume class equals forget concept |
| absent | `concept_aliases` | Supply from registry or an explicit empty list |
| `target_split` or split filename | `split` | Filename-derived values MUST be validated against content |
| absent | probe fields | Assign a registered legacy MCQ probe and `normal_image` only |
| `choices`, `answer_index` | same | Validate and preserve option order |
| absent | source/license/provenance | MUST be supplied by dataset config; missing values fail release validation |

Migration MUST write a report containing input/output hashes, migrated/rejected counts, every fallback used, and rejection reasons. Legacy records MUST NOT be silently mixed with v1 records. Migration does not invent aliases, licenses, matched-retain pairs, or provenance claims.

## 14. Conformance and change control

A `v1-conformant` release requires:

- schema validation with no ignored errors;
- split and answer-balance validation;
- frozen model, method, probe, prompt, scorer, and eligibility manifests;
- per-request records and required run artifacts;
- response-status decomposition;
- reproducible hashes and seed records;
- main results restricted to eligible concepts while reporting candidate coverage;
- no unsupported combination or oracle control in method averages;
- documentation consistent with executable behavior.

Normative changes after M0 MUST increment the specification version and add a Decision Log entry. Changes to full-study eligibility thresholds, prompt banks, attack budgets, scorers, or primary metrics after viewing method test results require a new benchmark version; they cannot be presented as the original preregistered protocol.

## 15. Mapping to planned code

| Specification object | Planned implementation |
|---|---|
| `BenchmarkItem`, `ProbeSpec`, `MethodSpec`, `ModelCapabilities`, `RunManifest` | `ulbench/schema.py` |
| model-specific gates and manifest | `ulbench/eligibility.py` |
| method-independent orchestration and capability errors | `ulbench/runner.py` |
| P0–P3 construction | `ulbench/probes/` |
| no-op, prompt suppression, output filter | `ulbench/methods/` |
| model capability adapters | `ulbench/models/` |
| access, leakage, retention, utility, cost, statistics | `ulbench/metrics/` |
| oracle and acquisition controls | `ulbench/audits/` |
| legacy JSONL migration and validation | `ulbench/tools/` or `scripts/` thin entry points |

The legacy `experiments/intext_unlearning.py` MAY remain as a compatibility wrapper, but new benchmark behavior MUST enter `ulbench/` and MUST NOT add more method-specific branches to the legacy runner.

## 16. Frozen M0 decisions and open pilot decisions

Frozen for v1 design:

- method-agnostic benchmark identity;
- R0/R1/R2 separation;
- main claims restricted to eligible, visually grounded, pre-existing behavioral knowledge;
- P0–P3 taxonomy and dual-format direct probing;
- separate refusal/invalid/error accounting;
- oracle prompts as controls only;
- G0 before full sweeps;
- per-item records and manifests as the source of paper numbers.

Open until the recorded G0 decision:

- final `τ_acc`, `τ_visual`, stability, and confidence-bound thresholds;
- retained concept axes and exact pilot models;
- final open-answer semantic scorer;
- fixed P3 attack bank and budget;
- depth-study R1/R2 implementations and general-utility suite versions.

These open choices do not authorize changing the claim boundary or using non-eligible concepts in main results.
