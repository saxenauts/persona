import requests
import uuid
from datetime import datetime
from .base import MemorySystem

class EverMemAdapter(MemorySystem):
    def __init__(self, base_url="http://localhost:1995"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1/memories"

    def add_session(self, user_id: str, session_data: str, date: str):
        # EverMem expects individual messages.
        # Format of session_data: "Role: Content\nRole: Content"
        lines = session_data.split('\n')
        
        # Parse date
        try:
            # Append generic time if only date provided
            base_time = f"{date}T00:00:00+00:00"
        except:
            base_time = datetime.utcnow().isoformat()

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            if ": " in line:
                role, content = line.split(": ", 1)
            else:
                role = "unknown"
                content = line
            
            # Construct a message ID
            msg_id = str(uuid.uuid4())
            
            payload = {
                "message_id": msg_id,
                "create_time": base_time, # ideally increment time slightly
                "sender": f"{user_id}_{role}", 
                "sender_name": role,
                "content": content,
                "group_id": user_id, # Use user_id as group_id for isolation
                "group_name": "Benchmark Session"
            }
            
            try:
                resp = requests.post(self.api_url, json=payload)
                resp.raise_for_status()
            except Exception as e:
                error_detail = "N/A"
                if hasattr(e, 'response') and e.response is not None:
                    error_detail = e.response.text
                print(f"[EverMem] Ingest fail: {e} | Detail: {error_detail}")

    def add_sessions(self, user_id: str, sessions: list):
        # Parallel ingest could overload it, but let's try strict sequential first to be safe with Docker
        for s in sessions:
            self.add_session(user_id, s['content'], s['date'])

    def query(self, user_id: str, query: str) -> str:
        # Search API: GET /api/v1/memories/search
        params = {
            "user_id": user_id, # Actually search uses user_id context logic usually
            "query": query,
            "retrieve_method": "hybrid"
        }
        # Note: EverMem might scope search by group_id in their logic?
        # The API doc said "Retrieve relevant memory data based on query text".
        # It takes 'user_id' parameter.
        
        resp = requests.get(f"{self.api_url}/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get('status') == 'ok':
            result = data.get('result', {})
            groups = result.get('groups', [])
            
            # Aggregate context
            context_pieces = []
            for g in groups:
                 for m in g.get('memories', []):
                     context_pieces.append(f"{m.get('timestamp')}: {m.get('content')}")
            
            context = "\n".join(context_pieces)
            
            # Now generate answer.
            # EverMem API just returns memories. We must generate the answer using our own LLM to be consistent with others?
            # Or does 'Cli' do generation? The README said "EverMemOS uses GPT-4.1-mini as answer LLM".
            # This implies the SYSTEM does it? 
            # But the Controller only has 'search'.
            # I will assume I need to Generate Answer myself using the retrieved context, similar to Mem0/Zep adapters.
            
            return self._generate_answer(query, context)
            
        return "No data"

    def _generate_answer(self, query, context):
        from langchain_openai import AzureChatOpenAI
        import os
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
            temperature=0,
        )
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        return llm.invoke(prompt).content

    def reset(self, user_id: str):
        # How to delete?
        # No API shown. Use direct Mongo delete?
        # Or just rely on unique user_id.
        pass
