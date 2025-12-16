import requests
import os
from .base import MemorySystem

class PersonaAdapter(MemorySystem):
    def __init__(self):
        self.base_url = "http://localhost:8000/api/v1"

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
                "metadata": {"date": date}
            },
            timeout=300  # Increased timeout for large batches
        )
        if response.status_code != 201:
            print(f"Persona Ingest Error: {response.status_code} - {response.text}")
        response.raise_for_status()

    def add_sessions(self, user_id: str, sessions: list):
        """
        Add multiple sessions using the new bulk ingestion endpoint.
        """
        # Create user if needed
        requests.post(f"{self.base_url}/users/{user_id}")
        
        # Prepare batch payload
        batch_items = []
        for s in sessions:
            content = f"Date: {s['date']}\n\n{s['content']}"
            batch_items.append({
                "title": f"Session {s['date']}",
                "content": content,
                "metadata": {"source": "benchmark"}
            })
            
        payload = {"batch": batch_items}
        
        try:
            resp = requests.post(f"{self.base_url}/users/{user_id}/ingest/batch", json=payload)
            resp.raise_for_status()
        except Exception as e:
            print(f"[PersonaAdapter] Batch ingest failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"  Response: {e.response.text}")
            raise e

    def query(self, user_id: str, query: str) -> str:
        # Benchmark usually asks for an answer.
        # Check API docs in README:
        # POST /api/v1/users/{user_id}/rag/query
        
        response = requests.post(
            f"{self.base_url}/users/{user_id}/rag/query",
            json={"query": query}
        )
        if response.status_code != 200:
            print(f"Persona Query Error: {response.status_code} - {response.text}")
        response.raise_for_status()
        # Expecting {"answer": "..."} or similar
        data = response.json()
        if isinstance(data, dict):
            return data.get("answer") or data.get("result") or str(data)
        return str(data)

    def reset(self, user_id: str):
        requests.delete(f"{self.base_url}/users/{user_id}")
