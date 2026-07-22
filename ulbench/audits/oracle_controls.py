"""Oracle instruction-following controls (roadmap B-P0.2, audit-only).

Migrated from the legacy ``ORACLE_HARD`` / ``ORACLE_REVERSE`` conditions in
``experiments/intext_unlearning.py::build_prompt``. Frozen decision #9
(HANDOUT.md): these measure instruction following, not unlearning, and must
never enter unlearning-method rankings. They therefore do NOT subclass
:class:`ulbench.methods.base.UnlearningMethod` — a runner cannot register
them as methods without a type error.

Instruction text must stay byte-identical to the legacy conditions;
``tests/test_methods.py`` enforces this.
"""

from __future__ import annotations

from ulbench.types import ProbeRequest


class OracleInstructionControl:
    """Per-item instruction control that reveals the ground-truth concept.

    Applies to the forget split only; legacy runs reuse baseline results on
    the retain split. ``transform_input`` follows the same shape as method
    input transforms so probe rendering stays uniform, but the class is
    intentionally outside the method hierarchy.
    """

    control_id: str = ""
    legacy_condition: str = ""
    semantic_label = "instruction_following_control"
    forget_split_only = True
    reveals_ground_truth = True
    _template: str = ""

    def instruction_line(self, target_concept: str) -> str:
        if not target_concept:
            raise ValueError(f"{self.control_id} requires the per-item target concept")
        return self._template.format(target=target_concept)

    def transform_input(self, request: ProbeRequest) -> ProbeRequest:
        if self.forget_split_only and request.context.get("split") not in (
            None, "test_forget", "train_forget",
        ):
            return request
        request.method_instructions.append(
            self.instruction_line(request.context.get("target_concept", ""))
        )
        return request

    def metadata(self) -> dict:
        return {
            "control_id": self.control_id,
            "legacy_condition": self.legacy_condition,
            "semantic_label": self.semantic_label,
            "forget_split_only": self.forget_split_only,
            "reveals_ground_truth": self.reveals_ground_truth,
        }


class OracleHardControl(OracleInstructionControl):
    """Upper-bound control: names the ground truth and forbids choosing it."""

    control_id = "oracle_hard"
    legacy_condition = "ORACLE_HARD"
    _template = (
        "The object in the image is {target}. "
        "If you see a {target}, you must not choose the correct option. "
        "Output one of the other indices."
    )


class OracleReverseControl(OracleInstructionControl):
    """Negation probe: asserts the image is NOT the target concept."""

    control_id = "oracle_reverse"
    legacy_condition = "ORACLE_REVERSE"
    _template = (
        "The object in this image is NOT a {target}. "
        "Choose the most likely answer from the remaining options."
    )


ORACLE_CONTROLS = {
    control.control_id: control
    for control in (OracleHardControl(), OracleReverseControl())
}
