"""Strict canonical JSON serialization for hashes, signatures, and bundles.

The accepted domain is the JSON data model, with tuples admitted as an
in-memory representation of JSON arrays.  In particular, this module never
stringifies mapping keys or arbitrary objects and never rewrites non-finite
numbers into strings.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import json
import math


class CanonicalJSONError(ValueError):
    """Raised when a value is outside the canonical JSON domain."""


def canonical_json_value(value: Any, *, _path: str = "$") -> Any:
    """Return a JSON-native value or reject unsupported/colliding types."""

    if value is None or type(value) in (str, bool):
        return value
    if type(value) is int:
        if abs(value) > 2**53 - 1:
            raise CanonicalJSONError(
                f"{_path} contains an integer outside the interoperable JSON range"
            )
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CanonicalJSONError(f"{_path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if type(key) is not str:
                raise CanonicalJSONError(
                    f"{_path} contains a non-string object key: {key!r}"
                )
            normalized[key] = canonical_json_value(item, _path=f"{_path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            canonical_json_value(item, _path=f"{_path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise CanonicalJSONError(
        f"{_path} contains unsupported canonical JSON type {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize *value* to the unique compact byte representation used by CIN."""

    normalized = canonical_json_value(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _reject_nonfinite_constant(value: str) -> None:
    raise CanonicalJSONError(f"non-finite JSON constant is not permitted: {value}")


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate JSON object key is not permitted: {key}")
        result[key] = value
    return result


def strict_json_loads(payload: str | bytes | bytearray) -> Any:
    """Parse JSON while rejecting duplicate keys and non-standard constants."""

    try:
        value = json.loads(
            payload,
            parse_constant=_reject_nonfinite_constant,
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except CanonicalJSONError:
        raise
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanonicalJSONError(f"invalid JSON: {error}") from error
    return canonical_json_value(value)
