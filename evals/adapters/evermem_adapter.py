import requests
import uuid
import json
import os
from datetime import datetime
from .base import MemorySystem

class EverMemAdapter(MemorySystem):
    def __init__(self, base_url="http://localhost:1995"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1/memories"
        # Stage logging for diagnosis
        self.last_stage_logs = {}
        self._create_log_dir()
    
    def _create_log_dir(self):
        """Create directory for detailed stage logs."""
        self.log_dir = "evals/results/evermem_stage_logs"
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _log_stage(self, user_id: str, stage: str, data: dict):
        """Log stage data for later diagnosis."""
        self.last_stage_logs[f"{user_id}_{stage}"] = data
        # Also write to file for persistence
        log_file = f"{self.log_dir}/{user_id}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps({"stage": stage, "timestamp": datetime.now().isoformat(), **data}) + "\n")

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

        messages_sent = 0
        messages_failed = 0
        
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
                resp = requests.post(self.api_url, json=payload, timeout=30)
                resp.raise_for_status()
                messages_sent += 1
            except Exception as e:
                messages_failed += 1
                error_detail = "N/A"
                if hasattr(e, 'response') and e.response is not None:
                    error_detail = e.response.text
                print(f"[EverMem] Ingest fail: {e} | Detail: {error_detail}")
        
        # Log stage 1 (ingestion) summary
        self._log_stage(user_id, "stage1_ingestion", {
            "date": date,
            "total_lines": len(lines),
            "messages_sent": messages_sent,
            "messages_failed": messages_failed,
        })

    def add_sessions(self, user_id: str, sessions: list):
        # Sequential ingest (EverMem relies on async server-side processing)
        print(f"    [EverMem] Ingesting {len(sessions)} sessions...")
        for s in sessions:
            self.add_session(user_id, s['content'], s['date'])

    def query(self, user_id: str, query: str) -> str:
        # Search API: GET /api/v1/memories/search
        params = {
            "user_id": user_id, # Actually search uses user_id context logic usually
            "query": query,
            "retrieve_method": "hybrid"
        }
        
        try:
            resp = requests.get(f"{self.api_url}/search", params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            error_msg = str(e)
            self._log_stage(user_id, "stage3_retrieval", {
                "query": query,
                "error": error_msg,
                "memories_count": 0
            })
            return f"Search error: {error_msg}"
        
        if data.get('status') == 'ok':
            result = data.get('result', {})
            memories = result.get('memories', [])
            
            # Aggregate context
            context_pieces = []
            for m in memories:
                context_pieces.append(f"{m.get('timestamp', 'N/A')}: {m.get('content', m.get('summary', ''))}")
            
            context = "\n".join(context_pieces)
            
            # Log stage 3 (retrieval) details
            self._log_stage(user_id, "stage3_retrieval", {
                "query": query,
                "memories_count": len(memories),
                "context_preview": context[:500] if context else "EMPTY",
                "raw_memories": memories[:5]  # First 5 for diagnosis
            })
            
            # Generate answer
            answer = self._generate_answer(query, context)
            
            # Log stage 4 (generation) details
            self._log_stage(user_id, "stage4_generation", {
                "query": query,
                "context_length": len(context),
                "answer": answer,
            })
            
            return answer
            
        # Log failed status
        self._log_stage(user_id, "stage3_retrieval", {
            "query": query,
            "status": data.get('status'),
            "error": data.get('message', 'Unknown error'),
            "memories_count": 0
        })
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
        # No API shown for deletion. Rely on unique user_id per question.
        pass
    
    def get_stage_logs(self, user_id: str) -> dict:
        """Return stage logs for diagnosis."""
        return {k: v for k, v in self.last_stage_logs.items() if k.startswith(user_id)}
