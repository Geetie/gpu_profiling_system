"""Tool Runner — wires validation, approval, and execution.

The ToolRunner.execute() method is the callable passed to
AgentLoop.set_tool_executor(). It implements the full pipeline:
validate input → check approval → execute handler → validate output → persist.
"""
from __future__ import annotations

from typing import Any, Callable

from src.application.approval_queue import ApprovalQueue, ApprovalRequest, ApprovalStatus
from src.domain.permission import PermissionChecker
from src.domain.schema_validator import SchemaValidator, SchemaValidationError
from src.domain.tool_contract import ToolRegistry
from src.infrastructure.state_persist import StatePersister


class ApprovalRequiredError(Exception):
    """Raised when tool execution requires human approval.

    The caller should present the request to the user, respond to the
    approval queue, then re-call execute().
    """

    def __init__(self, request: ApprovalRequest) -> None:
        self.request = request
        super().__init__(
            f"Tool '{request.tool_name}' requires approval "
            f"(request_id={request.id})"
        )


class ToolRunner:
    """Full tool execution pipeline.

    Wires together:
    1. Tool lookup (P2 fail-closed)
    2. Input schema validation
    3. Approval checking
    4. Handler execution
    5. Output schema validation
    6. State persistence (P6)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        tool_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
        approval_queue: ApprovalQueue,
        permission_checker: PermissionChecker,
        persister: StatePersister,
        validator: SchemaValidator | None = None,
    ) -> None:
        self._registry = registry
        self._handlers = tool_handlers
        self._approval_queue = approval_queue
        self._permission_checker = permission_checker
        self._persister = persister
        self._validator = validator or SchemaValidator()

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool through the full pipeline.

        Args:
            tool_name: Registered tool name.
            arguments: Tool input arguments (must match input_schema).

        Returns:
            Tool output dict (matches output_schema).

        Raises:
            KeyError: if tool not registered (P2).
            SchemaValidationError: if input or output doesn't match schema.
            PermissionError: if permission denied.
            ApprovalRequiredError: if human approval is needed.
        """
        # Step 1: Lookup tool (P2 fail-closed)
        contract = self._registry.get(tool_name)

        # Step 2: Validate input schema (returns coerced data)
        arguments = self._validator.validate(contract.input_schema, arguments)

        # Step 3: Check approval requirements
        needs_approval = False
        if contract.requires_approval:
            for perm in contract.permissions:
                if self._permission_checker.requires_approval(perm):
                    needs_approval = True
                    break

        if needs_approval:
            request = self._approval_queue.submit(
                tool_name=tool_name,
                arguments=arguments,
                permissions=contract.permissions,
                mode=self._permission_checker.mode,
            )
            if request.status == ApprovalStatus.PENDING:
                raise ApprovalRequiredError(request)
            elif request.status in (ApprovalStatus.REJECTED, ApprovalStatus.AUTO_REJECTED):
                raise PermissionError(
                    f"Tool '{tool_name}' approval {request.status.value}: "
                    f"{request.reason}"
                )

        # Step 4: Execute handler
        handler = self._handlers.get(tool_name)
        if handler is None:
            self._persister.log_error(
                error_type="NoToolHandler",
                context=f"tool:{tool_name}",
                message=f"No handler registered for tool '{tool_name}'",
            )
            return {
                "error": f"No handler for tool '{tool_name}'",
                "status": "no_handler",
            }

        result = handler(arguments)

        # Step 5: Validate output schema (returns coerced data)
        try:
            result = self._validator.validate(contract.output_schema, result)
        except SchemaValidationError:
            self._persister.log_error(
                error_type="SchemaValidationError",
                context=f"tool:{tool_name}",
                message=f"Output validation failed for '{tool_name}'",
            )
            raise

        # Step 6: Persist (P6)
        # Determine status: check explicit success/success field, then error field
        status = self._determine_status(result)
        self._persister.log_tool_execution(
            tool_name=tool_name,
            inputs=arguments,
            status=status,
        )

        return result

    @staticmethod
    def _determine_status(result: dict[str, Any]) -> str:
        """Determine execution status from result dict.

        Checks explicit fields first (success, status), then falls back
        to error field heuristics.
        """
        # Check explicit success field
        if "success" in result:
            return "success" if result["success"] else "failed"
        # Check explicit status field
        if "status" in result:
            return str(result["status"])
        # Check error field — only treat non-empty string as failure
        error = result.get("error")
        if isinstance(error, str) and error:
            return "completed_with_errors"
        return "success"
