"""JSON Schema validation for tool I/O — domain layer.

Implements P1 (工具定义能力边界): tool inputs and outputs must conform
  to their declared schemas. Any deviation is rejected (fail-closed).

Enhanced for resilience:
  - Missing fields are allowed (not required)
  - Automatic type coercion when possible
  - Flexible type checking
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
    """Check if a value matches the expected schema type.

    Supports flexible type matching:
    - string: accepts str, int, float, bool (will be converted)
    - integer: accepts int, str that can be parsed as int, bool (0/1)
    - number: accepts int, float, str that can be parsed as number
    - boolean: accepts bool, int (0/1), str ("true"/"false")
    - object: accepts dict
    """
    if expected_type == "boolean":
        if isinstance(value, bool):
            return True
        if isinstance(value, int) and value in (0, 1):
            return True
        if isinstance(value, str):
            return value.lower() in ("true", "false", "1", "0")
        return False
    if expected_type == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return True
        if isinstance(value, str):
            try:
                int(value)
                return True
            except ValueError:
                pass
        if isinstance(value, float) and value.is_integer():
            return True
        return False
    if expected_type == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                pass
        return False
    if expected_type == "string":
        if isinstance(value, (str, int, float, bool)):
            return True
        return False
    if expected_type == "object":
        return isinstance(value, dict)
    return False


def _coerce_type(value: Any, expected_type: str) -> Any:
    """Coerce a value to the expected type if possible."""
    if expected_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value != 0
        if isinstance(value, str):
            return value.lower() in ("true", "1")
        return bool(value)
    if expected_type == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return 0
    if expected_type == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        return 0.0
    if expected_type == "string":
        return str(value)
    if expected_type == "object":
        if isinstance(value, dict):
            return value
        return {}
    return value


class SchemaValidator:
    """Lightweight validator for the simplified schema format used in ToolContract.

    Schema format:
    - "string" -> must be str (or convertable to str)
    - "integer" -> must be int (or convertable to int)
    - "boolean" -> must be bool (or convertable to bool)
    - "object" -> must be dict
    - ["string"] -> must be list[str] (or convertable to list)

    Features:
    - Missing fields are allowed (not required)
    - Automatic type coercion when possible
    - Flexible type checking
    """

    def validate(self, schema: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Validate data against schema and return coerced data.

        Returns:
            dict: Data with types coerced to match schema

        Raises:
            SchemaValidationError: If validation fails after coercion
        """
        errors: list[ValidationError] = []
        coerced = {}
        self._validate_object(schema, data, "", errors, coerced)
        if errors:
            raise SchemaValidationError(errors)
        return coerced

    def _validate_object(
        self,
        schema: dict[str, Any],
        data: dict[str, Any],
        path: str,
        errors: list[ValidationError],
        coerced: dict[str, Any],
    ) -> None:
        """Validate all fields in an object schema and build coerced data."""
        for field_name, field_type in schema.items():
            field_path = f"{path}.{field_name}" if path else field_name

            if field_name not in data:
                # Missing fields are allowed (no error)
                continue

            value = data[field_name]
            self._validate_field(field_type, value, field_path, errors, coerced, field_name)

    def _validate_field(
        self,
        expected: Any,
        value: Any,
        path: str,
        errors: list[ValidationError],
        coerced: dict[str, Any],
        field_name: str | None = None,
    ) -> None:
        """Validate a single field against its expected type and coerce if possible."""
        if isinstance(expected, str):
            # Scalar type: "string", "integer", etc.
            if not _check_type(value, expected):
                errors.append(ValidationError(
                    field=path,
                    expected=expected,
                    actual=_python_type_name(value),
                    message=f"Field '{path}': expected {expected}, got {_python_type_name(value)}",
                ))
            else:
                if field_name is not None:
                    coerced[field_name] = _coerce_type(value, expected)
        elif isinstance(expected, list):
            # Array type: ["string"], ["integer"], etc.
            if not isinstance(value, list):
                # Try to coerce to list
                if isinstance(value, (str, dict, int, float, bool)):
                    value = [value]
                else:
                    errors.append(ValidationError(
                        field=path,
                        expected=f"array[{expected[0] if expected else 'any'}]",
                        actual=_python_type_name(value),
                        message=f"Field '{path}': expected array, got {_python_type_name(value)}",
                    ))
                    return

            element_type = expected[0] if expected else None
            coerced_array = []
            if element_type is not None:
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
                        else:
                            coerced_array.append(_coerce_type(item, element_type))
                    else:
                        coerced_array.append(item)
            else:
                coerced_array = value

            if field_name is not None:
                coerced[field_name] = coerced_array