"""Tests for JSON Schema validator (domain/schema_validator.py)."""
import pytest
from src.domain.schema_validator import (
    SchemaValidator,
    SchemaValidationError,
    ValidationError,
)


class TestSchemaValidator:
    def test_valid_string_field(self):
        validator = SchemaValidator()
        schema = {"name": "string"}
        validator.validate(schema, {"name": "hello"})

    def test_valid_integer_field(self):
        validator = SchemaValidator()
        schema = {"count": "integer"}
        validator.validate(schema, {"count": 42})

    def test_valid_boolean_field(self):
        validator = SchemaValidator()
        schema = {"flag": "boolean"}
        validator.validate(schema, {"flag": True})

    def test_valid_object_field(self):
        validator = SchemaValidator()
        schema = {"config": "object"}
        validator.validate(schema, {"config": {"key": "value"}})

    def test_valid_array_field(self):
        validator = SchemaValidator()
        schema = {"items": ["string"]}
        validator.validate(schema, {"items": ["a", "b", "c"]})

    def test_missing_required_field(self):
        validator = SchemaValidator()
        schema = {"name": "string"}
        with pytest.raises(SchemaValidationError) as exc:
            validator.validate(schema, {})
        assert "name" in str(exc.value)

    def test_wrong_scalar_type(self):
        validator = SchemaValidator()
        schema = {"count": "integer"}
        with pytest.raises(SchemaValidationError):
            validator.validate(schema, {"count": "not_a_number"})

    def test_bool_not_treated_as_int(self):
        """Boolean should NOT pass integer validation."""
        validator = SchemaValidator()
        schema = {"count": "integer"}
        with pytest.raises(SchemaValidationError):
            validator.validate(schema, {"count": True})

    def test_int_not_treated_as_bool(self):
        """Integer should NOT pass boolean validation."""
        validator = SchemaValidator()
        schema = {"flag": "boolean"}
        with pytest.raises(SchemaValidationError):
            validator.validate(schema, {"flag": 1})

    def test_wrong_array_element_type(self):
        validator = SchemaValidator()
        schema = {"items": ["string"]}
        with pytest.raises(SchemaValidationError):
            validator.validate(schema, {"items": ["ok", 42, "fine"]})

    def test_array_instead_of_string(self):
        validator = SchemaValidator()
        schema = {"name": "string"}
        with pytest.raises(SchemaValidationError):
            validator.validate(schema, {"name": ["not", "a", "string"]})

    def test_extra_fields_allowed(self):
        """Extra fields in data should be allowed (lenient)."""
        validator = SchemaValidator()
        schema = {"name": "string"}
        validator.validate(schema, {"name": "hello", "extra": "field"})

    def test_multiple_errors_collected(self):
        """All errors should be collected, not just the first one."""
        validator = SchemaValidator()
        schema = {"a": "string", "b": "integer"}
        with pytest.raises(SchemaValidationError) as exc:
            validator.validate(schema, {"a": 123, "b": "not_int"})
        assert len(exc.value.errors) == 2

    def test_empty_schema_passes(self):
        validator = SchemaValidator()
        validator.validate({}, {})
        validator.validate({}, {"extra": "data"})

    def test_nested_object_as_object_type(self):
        """object type should accept any dict value."""
        validator = SchemaValidator()
        schema = {"config": "object"}
        validator.validate(schema, {"config": {"a": 1, "b": [1, 2, 3]}})

    def test_float_for_number_type(self):
        validator = SchemaValidator()
        schema = {"value": "number"}
        validator.validate(schema, {"value": 3.14})

    def test_int_for_number_type(self):
        """Integer should also pass number type."""
        validator = SchemaValidator()
        schema = {"value": "number"}
        validator.validate(schema, {"value": 42})
