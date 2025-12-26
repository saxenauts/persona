from abc import ABC, abstractmethod


class MemorySystem(ABC):
    log_node_content: bool = False

    @abstractmethod
    def add_session(self, user_id: str, session_data: str, date: str):
        """
        Ingest a session into the memory system.

        Args:
            user_id: Unique identifier for the user.
            session_data: The text content of the session/conversation.
            date: The date of the session (YYYY-MM-DD) for temporal context.
        """
        pass

    def add_sessions(self, user_id: str, sessions: list):
        """
        Bulk ingest sessions. Default implementation loops.
        Adapters should override this for optimization.

        Args:
            user_id: User ID.
            sessions: List of dicts [{"content": str, "date": str}]
        """
        for s in sessions:
            self.add_session(user_id, s["content"], s["date"])

    @abstractmethod
    def query(self, user_id: str, query: str) -> str:
        """
        Query the memory system.

        Args:
            user_id: Unique identifier for the user.
            query: The question to ask.

        Returns:
            The answer string.
        """
        pass

    @abstractmethod
    def reset(self, user_id: str):
        """
        Clear memory for a specific user to ensure clean state for benchmarks.
        """
        pass
