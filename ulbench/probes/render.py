"""Prompt rendering for v1 probes (runner-side, layout-owning).

Methods inject instruction lines via ``ProbeRequest.method_instructions``;
this renderer decides their placement, so methods stay independent of prompt
layout. The MCQ layout is byte-identical to the legacy
``experiments.intext_unlearning.build_prompt`` output (enforced by
``tests/test_render.py``) so new-runner numbers stay comparable to legacy
runs.
"""

from __future__ import annotations

from typing import Iterable

from ulbench.schema import BenchmarkItem, QuestionFormat

RENDERER_VERSION = "prompt_render.v1"

SHORT_ANSWER_DIRECTIVE = (
    "Reply with the short name only — nothing else.\n"
    "Answer:"
)


def _digit_run(count: int) -> str:
    digits = [str(index) for index in range(count)]
    return ", ".join(digits[:-1]) + f", or {digits[-1]}"


def render_prompt(item: BenchmarkItem,
                  method_instructions: Iterable[str] = ()) -> str:
    """Render the final text prompt for *item* plus method instructions."""
    if item.question_format == QuestionFormat.MCQ:
        lines: list[str] = []
        if item.question.strip():
            lines += [f"Q: {item.question}", ""]
        for index, choice in enumerate(item.choices):
            lines.append(f"{index}) {choice}")
        lines.append("")
        for instruction in method_instructions:
            lines.append(instruction)
            lines.append("")
        lines.append(
            f"Reply with a single digit {_digit_run(len(item.choices))} — "
            "nothing else.\n"
            "Answer:"
        )
        return "\n".join(lines)

    if item.question_format == QuestionFormat.SHORT_ANSWER:
        lines = [f"Q: {item.question}", ""]
        for instruction in method_instructions:
            lines.append(instruction)
            lines.append("")
        lines.append(SHORT_ANSWER_DIRECTIVE)
        return "\n".join(lines)

    raise ValueError(
        f"{item.item_id}: no renderer for question format "
        f"{item.question_format.value}"
    )
