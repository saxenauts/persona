import requests
import os
import re
from .base import MemorySystem


def extract_question_for_retrieval(full_query: str) -> str:
    """Extract just the question part for retrieval, stripping MCQ options.

    PersonaMem format:
    Question: <the actual question>
    Options: (a)... (b)... (c)... (d)...
    Answer with only the letter (a/b/c/d).

    We want ONLY the question for retrieval to avoid entity contamination.
    """
    if "Options:" in full_query:
        parts = full_query.split("Options:")
        question_part = parts[0].strip()
        if question_part.startswith("Question:"):
            question_part = question_part[len("Question:") :].strip()
        return question_part
    return full_query


class PersonaAdapter(MemorySystem):
    def __init__(self):
        port = os.environ.get("PERSONA_PORT", "8000")
        self.base_url = f"http://localhost:{port}/api/v1"
        self.last_ingest_stats = None
        self.last_query_stats = None

    def add_session(self, user_id: str, session_data: str, date: str):
        # First ensure user exists
        resp = requests.post(f"{self.base_url}/users/{user_id}")
        if resp.status_code not in [200, 201]:
            print(f"Persona Create User Error: {resp.status_code} - {resp.text}")
            resp.raise_for_status()

        # Ingest
        response = requests.post(
            f"{self.base_url}/users/{user_id}/ingest",
            json={
                "title": f"Session {date}",
                "content": session_data,
                "metadata": {"date": date},
            },
            timeout=300,  # Increased timeout for large batches
        )
        if response.status_code != 201:
            print(f"Persona Ingest Error: {response.status_code} - {response.text}")
        response.raise_for_status()
        try:
            self.last_ingest_stats = response.json()
        except ValueError:
            self.last_ingest_stats = None

    def add_sessions(self, user_id: str, sessions: list):
        """
        Add multiple sessions using the new bulk ingestion endpoint.
        Uses IngestBatchRequest format with 'items' array.
        """
        # Create user if needed
        requests.post(f"{self.base_url}/users/{user_id}")

        # Prepare batch payload using new API format
        items = []
        for s in sessions:
            content = f"Date: {s['date']}\n\n{s['content']}"
            items.append({"content": content, "source_type": "conversation"})

        payload = {"items": items}

        try:
            print(
                f"      → Calling /ingest/batch with {len(items)} items...", flush=True
            )
            resp = requests.post(
                f"{self.base_url}/users/{user_id}/ingest/batch",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            print(f"      → Batch ingest complete", flush=True)
            try:
                self.last_ingest_stats = resp.json()
            except ValueError:
                self.last_ingest_stats = None
        except Exception as e:
            print(f"[PersonaAdapter] Batch ingest failed: {e}")
            if hasattr(e, "response") and e.response:
                print(f"  Response: {e.response.text}")
            raise e

    def query(self, user_id: str, query: str) -> str:
        retrieval_query = extract_question_for_retrieval(query)

        response = requests.post(
            f"{self.base_url}/users/{user_id}/rag/query",
            json={
                "query": query,
                "retrieval_query": retrieval_query,
                "include_stats": True,
            },
        )
        if response.status_code != 200:
            print(f"Persona Query Error: {response.status_code} - {response.text}")
        response.raise_for_status()
        # Expecting {"answer": "..."} or similar
        data = response.json()
        if isinstance(data, dict):
            if "answer" in data:
                self.last_query_stats = data
                return data.get("answer")
            return data.get("result") or str(data)
        return str(data)

    def reset(self, user_id: str):
        requests.delete(f"{self.base_url}/users/{user_id}")
