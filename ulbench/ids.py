"""Deterministic identifier helpers shared by construction and migration code.

Every derived artifact (migrated item, control variant, reordered MCQ) must
reproduce the same IDs from the same inputs, so all ID construction funnels
through these two functions.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized or hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def stable_id(prefix: str, *parts: Any) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()[:20]}"


def option_order_hash(choices: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(choices, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
