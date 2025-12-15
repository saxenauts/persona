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
            }
        )
        if response.status_code != 201:
            print(f"Persona Ingest Error: {response.status_code} - {response.text}")
        response.raise_for_status()

    def add_sessions(self, user_id: str, sessions: list):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # 1. Ensure user exists (once)
        resp = requests.post(f"{self.base_url}/users/{user_id}")
        if resp.status_code not in [200, 201]:
             resp.raise_for_status()

        # 2. Parallel Ingest
        def _post(session):
            return requests.post(
                f"{self.base_url}/users/{user_id}/ingest",
                json={
                    "title": f"Session {session['date']}",
                    "content": session['content'],
                    "metadata": {"date": session['date']}
                }
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_post, s) for s in sessions]
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res.status_code != 201:
                        print(f"  ! Persona Ingest Fail: {res.text}")
                except Exception as e:
                    print(f"  ! Persona Thread Fail: {e}")

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
