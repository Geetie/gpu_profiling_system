"""Typed EventBus — centralized event dispatch with filtering and priority.

Observer pattern: decouples event producers from event consumers.
Agents, pipeline stages, and infrastructure emit events through the bus;
UI, logging, and audit components subscribe to specific event types.

Design goals:
1. Type-safe event dispatch (EventKind enum)
2. Priority ordering (higher priority handlers run first)
3. Filtering (subscribe to specific event kinds)
4. Async-ready (synchronous now, async-ready structure)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventKind(Enum):
    START = "start"
    TURN = "turn"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    COMPRESS = "compress"
    STOP = "stop"
    ERROR = "error"
    PERSIST = "persist"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"


@dataclass
class LoopEvent:
    kind: EventKind
    payload: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[LoopEvent], None]


@dataclass
class _Subscription:
    handler: EventHandler
    kinds: set[EventKind] | None
    priority: int


class EventBus:
    """Centralized event dispatch with filtering and priority.

    Usage:
        bus = EventBus()

        # Subscribe to all events
        bus.subscribe(my_handler)

        # Subscribe to specific event kinds
        bus.subscribe(my_handler, kinds={EventKind.TOOL_CALL, EventKind.TOOL_RESULT})

        # Subscribe with priority (higher = runs first)
        bus.subscribe(audit_handler, kinds={EventKind.ERROR}, priority=100)

        # Emit events
        bus.emit(LoopEvent(kind=EventKind.TOOL_CALL, payload={"tool": "compile_cuda"}))
    """

    def __init__(self) -> None:
        self._subscriptions: list[_Subscription] = []

    def subscribe(
        self,
        handler: EventHandler,
        kinds: set[EventKind] | None = None,
        priority: int = 0,
    ) -> None:
        """Register an event handler.

        Args:
            handler: Callback function receiving LoopEvent.
            kinds: If None, receives all events. Otherwise only listed kinds.
            priority: Higher priority handlers execute first. Default 0.
        """
        self._subscriptions.append(_Subscription(
            handler=handler,
            kinds=kinds,
            priority=priority,
        ))
        self._subscriptions.sort(key=lambda s: s.priority, reverse=True)

    def emit(self, event: LoopEvent) -> None:
        """Dispatch an event to all matching subscribers.

        Handlers are called in priority order (highest first).
        Exceptions in handlers are caught and logged to prevent
        one failing handler from blocking others.
        """
        for sub in self._subscriptions:
            if sub.kinds is not None and event.kind not in sub.kinds:
                continue
            try:
                sub.handler(event)
            except Exception as e:
                print(f"[EventBus] Handler {sub.handler.__name__} raised: {e}")

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._subscriptions.clear()

    @property
    def subscription_count(self) -> int:
        return len(self._subscriptions)
