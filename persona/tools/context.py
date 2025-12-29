"""Tool execution context - injected at runtime, not exposed to LLM."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from persona.core.graph_ops import GraphOps
    from persona.core.memory_store import MemoryStore
    from persona.models.memory import UserCard


@dataclass
class ToolContext:
    """
    Runtime context for tool execution.

    This is the "hippocampal index" - the unified interface between
    LLM tool calls and the memory graph. LLM provides structured
    parameters (dates, types, filters); we execute against the graph.

    Not exposed to LLM - injected by the runner at execution time.
    """

    user_id: str
    graph_ops: "GraphOps"
    store: "MemoryStore"
    session_id: Optional[str] = None
    user_timezone: str = "UTC"
    user_card: Optional["UserCard"] = None

    # Execution tracking (for bounded parallel work)
    _subtasks: list[dict[str, Any]] = field(default_factory=list)

    def track_subtask(self, name: str, status: str = "pending") -> int:
        """Track a subtask for progress reporting. Returns subtask index."""
        idx = len(self._subtasks)
        self._subtasks.append({"name": name, "status": status, "result": None})
        return idx

    def complete_subtask(
        self, idx: int, status: str = "done", result: Any = None
    ) -> None:
        """Mark subtask complete."""
        if 0 <= idx < len(self._subtasks):
            self._subtasks[idx]["status"] = status
            self._subtasks[idx]["result"] = result

    @property
    def subtask_summary(self) -> dict[str, int]:
        """Summary of subtask statuses."""
        counts: dict[str, int] = {}
        for t in self._subtasks:
            counts[t["status"]] = counts.get(t["status"], 0) + 1
        return counts
