"""JSON Schema validation for tool I/O — domain layer.

Implements P1 (工具定义能力边界): tool inputs and outputs must conform
to their declared schemas. Any deviation is rejected (fail-closed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ── Validation Errors ────────────────────────────────────────────────


@dataclass
class ValidationError:
    """A single schema violation."""
    field: str
    expected: str
    actual: str
    message: str


class SchemaValidationError(Exception):
    """Raised when data does not match the tool's schema."""

    def __init__(self, errors: list[ValidationError]) -> None:
        self.errors = errors
        msgs = "; ".join(e.message for e in errors)
        super().__init__(f"Schema validation failed: {msgs}")


# ── Schema Validator ─────────────────────────────────────────────────


# Type mapping from simplified schema strings to Python types
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "boolean": bool,
    "object": dict,
    "number": (int, float),
}


def _python_type_name(value: Any) -> str:
    """Return the schema-style type name for a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _check_type(value: Any, expected_type: str) -> bool:
    """Check if a value matches the expected schema type."""
    if expected_type == "boolean":
        # In JSON, boolean is distinct from int; in Python bool is subclass of int
        return isinstance(value, bool)
    if expected_type == "integer":
        # Accept int but reject bool (since bool is subclass of int in Python)
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type in _TYPE_MAP:
        expected = _TYPE_MAP[expected_type]
        if isinstance(expected, tuple):
            return isinstance(value, expected)
        return isinstance(value, expected)
    return False


class SchemaValidator:
    """Lightweight validator for the simplified schema format used in ToolContract.

    Schema format:
    - "string" -> must be str
    - "integer" -> must be int (not bool)
    - "boolean" -> must be bool
    - "object" -> must be dict
    - ["string"] -> must be list[str]
    """

    def validate(self, schema: dict[str, Any], data: dict[str, Any]) -> None:
        """Validate data against schema. Raises SchemaValidationError on mismatch."""
        errors: list[ValidationError] = []
        self._validate_object(schema, data, "", errors)
        if errors:
            raise SchemaValidationError(errors)

    def _validate_object(
        self,
        schema: dict[str, Any],
        data: dict[str, Any],
        path: str,
        errors: list[ValidationError],
    ) -> None:
        """Validate all fields in an object schema."""
        for field_name, field_type in schema.items():
            field_path = f"{path}.{field_name}" if path else field_name

            if field_name not in data:
                errors.append(ValidationError(
                    field=field_path,
                    expected=field_type if isinstance(field_type, str) else str(field_type),
                    actual="missing",
                    message=f"Missing required field '{field_path}'",
                ))
                continue

            value = data[field_name]
            self._validate_field(field_type, value, field_path, errors)

    def _validate_field(
        self,
        expected: Any,
        value: Any,
        path: str,
        errors: list[ValidationError],
    ) -> None:
        """Validate a single field against its expected type."""
        if isinstance(expected, str):
            # Scalar type: "string", "integer", etc.
            if not _check_type(value, expected):
                errors.append(ValidationError(
                    field=path,
                    expected=expected,
                    actual=_python_type_name(value),
                    message=f"Field '{path}': expected {expected}, got {_python_type_name(value)}",
                ))
        elif isinstance(expected, list):
            # Array type: ["string"], ["integer"], etc.
            if not isinstance(value, list):
                errors.append(ValidationError(
                    field=path,
                    expected=f"array[{expected[0] if expected else 'any'}]",
                    actual=_python_type_name(value),
                    message=f"Field '{path}': expected array, got {_python_type_name(value)}",
                ))
                return

            element_type = expected[0] if expected else None
            if element_type is None:
                return  # Unconstrained array, accept anything

            for i, item in enumerate(value):
                item_path = f"{path}[{i}]"
                if isinstance(element_type, str):
                    if not _check_type(item, element_type):
                        errors.append(ValidationError(
                            field=item_path,
                            expected=element_type,
                            actual=_python_type_name(item),
                            message=f"Field '{item_path}': expected {element_type}, got {_python_type_name(item)}",
                        ))
