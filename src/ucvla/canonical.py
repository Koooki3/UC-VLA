# SPDX-License-Identifier: Apache-2.0
"""Canonical JSON, hashing and a version-independent deterministic RNG.

Everything that is hashed, serialised or sampled in M1 goes through this module so that
artifacts are byte-reproducible across machines and Python versions.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
from collections.abc import Mapping, MutableSequence, Sequence
from hashlib import sha256
from typing import Any, TypeVar

T = TypeVar("T")


def to_jsonable(obj: Any) -> Any:
    """Convert dataclasses / enums / tuples to plain JSON types; reject anything ambiguous."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value)
    if isinstance(obj, Mapping):
        if not all(isinstance(k, str) for k in obj):
            raise TypeError("canonical JSON requires string keys")
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(to_jsonable(v) for v in obj)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("NaN/inf not allowed in canonical JSON")
        return obj
    raise TypeError(f"not canonically serialisable: {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    return json.dumps(
        to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def sha256_hex(data: bytes) -> str:
    return sha256(data).hexdigest()


def digest(obj: Any) -> str:
    """sha256 of the canonical JSON encoding of ``obj``."""
    return sha256_hex(canonical_json(obj).encode("ascii"))


def is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


class HashRng:
    """Counter-mode sha256 PRNG: identical streams on every platform and Python version.

    Keys are strings, so an item's randomness depends only on its own key, not on how many
    other items were generated before it.
    """

    def __init__(self, *key: object) -> None:
        self._key = "|".join(str(k) for k in key).encode("utf-8")
        self._counter = 0

    def _next64(self) -> int:
        block = sha256(self._key + b"#" + str(self._counter).encode("ascii")).digest()
        self._counter += 1
        return int.from_bytes(block[:8], "big")

    def below(self, n: int) -> int:
        """Uniform integer in [0, n) via rejection sampling (no modulo bias)."""
        if n <= 0:
            raise ValueError("n must be positive")
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            x = self._next64()
            if x < limit:
                return x % n

    def randint(self, lo: int, hi: int) -> int:
        """Uniform integer in [lo, hi] inclusive."""
        if hi < lo:
            raise ValueError("empty range")
        return lo + self.below(hi - lo + 1)

    def choice(self, seq: Sequence[T]) -> T:
        return seq[self.below(len(seq))]

    def shuffle(self, seq: MutableSequence[Any]) -> None:
        for i in range(len(seq) - 1, 0, -1):
            j = self.below(i + 1)
            seq[i], seq[j] = seq[j], seq[i]

    def coin(self) -> bool:
        return self.below(2) == 1
