import os
from typing import Optional
from .base import MemorySystem


class HonchoAdapter(MemorySystem):
    """
    Adapter for Honcho memory system (https://github.com/plastic-labs/honcho).

    Honcho uses a peer-based model with sessions containing messages.
    Key features: dialectic endpoint, working representations, context retrieval.

    Environment variables:
        HONCHO_API_KEY: API key for Honcho cloud (or self-hosted)
        HONCHO_BASE_URL: Optional base URL for self-hosted instances
    """

    def __init__(self):
        try:
            from honcho import Honcho
        except ImportError:
            raise ImportError("honcho-ai not installed. Run: pip install honcho-ai")

        api_key = os.environ.get("HONCHO_API_KEY")
        base_url = os.environ.get("HONCHO_BASE_URL")

        if not api_key:
            raise ValueError("HONCHO_API_KEY environment variable required")

        self.client = (
            Honcho(api_key=api_key, base_url=base_url)
            if base_url
            else Honcho(api_key=api_key)
        )
        self._sessions = {}
        self._user_peers = {}
        self._assistant_peers = {}
        self.last_ingest_stats = None
        self.last_query_stats = None

    def _get_or_create_user_peer(self, user_id: str):
        if user_id not in self._user_peers:
            self._user_peers[user_id] = self.client.peer(f"user-{user_id}")
        return self._user_peers[user_id]

    def _get_or_create_assistant_peer(self, user_id: str):
        if user_id not in self._assistant_peers:
            self._assistant_peers[user_id] = self.client.peer(f"assistant-{user_id}")
        return self._assistant_peers[user_id]

    def _get_or_create_session(self, user_id: str, session_id: str):
        key = f"{user_id}:{session_id}"
        if key not in self._sessions:
            session = self.client.session(session_id)
            user_peer = self._get_or_create_user_peer(user_id)
            assistant_peer = self._get_or_create_assistant_peer(user_id)
            session.add_peers([user_peer, assistant_peer])
            self._sessions[key] = session
        return self._sessions[key]

    def add_session(self, user_id: str, session_data: str, date: str):
        session_id = f"{user_id}-{date}"
        session = self._get_or_create_session(user_id, session_id)
        user_peer = self._get_or_create_user_peer(user_id)

        messages = []
        for line in session_data.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("User:") or line.startswith("Human:"):
                content = line.split(":", 1)[1].strip() if ":" in line else line
                messages.append(user_peer.message(content, metadata={"date": date}))
            elif line.startswith("Assistant:") or line.startswith("AI:"):
                content = line.split(":", 1)[1].strip() if ":" in line else line
                assistant_peer = self._get_or_create_assistant_peer(user_id)
                messages.append(
                    assistant_peer.message(content, metadata={"date": date})
                )
            else:
                messages.append(user_peer.message(line, metadata={"date": date}))

        if messages:
            session.add_messages(messages)

        self.last_ingest_stats = {
            "messages_added": len(messages),
            "session_id": session_id,
        }

    def add_sessions(self, user_id: str, sessions: list):
        total_messages = 0
        for s in sessions:
            self.add_session(user_id, s["content"], s["date"])
            if self.last_ingest_stats:
                total_messages += self.last_ingest_stats.get("messages_added", 0)

        self.last_ingest_stats = {
            "total_messages": total_messages,
            "sessions_count": len(sessions),
        }

    def query(self, user_id: str, query: str) -> str:
        user_peer = self._get_or_create_user_peer(user_id)
        assistant_peer = self._get_or_create_assistant_peer(user_id)

        try:
            response = assistant_peer.chat(query)
            self.last_query_stats = {"query": query, "response_length": len(response)}
            return response
        except Exception as e:
            context_parts = []
            for key, session in self._sessions.items():
                if key.startswith(f"{user_id}:"):
                    try:
                        ctx = session.get_context(summary=True, tokens=2000)
                        if ctx:
                            context_parts.append(str(ctx))
                    except Exception:
                        pass

            if context_parts:
                combined_context = "\n\n".join(context_parts)
                return (
                    f"Based on context: {combined_context[:1000]}... (query: {query})"
                )

            return f"Error querying Honcho: {e}"

    def reset(self, user_id: str):
        keys_to_remove = [k for k in self._sessions if k.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del self._sessions[key]

        if user_id in self._user_peers:
            del self._user_peers[user_id]
        if user_id in self._assistant_peers:
            del self._assistant_peers[user_id]

        self.last_ingest_stats = None
        self.last_query_stats = None
