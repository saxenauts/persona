import os
import json
import time
from datetime import datetime
from evals.adapters.persona_adapter import PersonaAdapter
from evals.adapters.mem0_adapter import Mem0Adapter
from evals.adapters.zep_adapter import ZepAdapter
from langchain_openai import AzureChatOpenAI

class BenchmarkRunner:
    def __init__(self):
        self.adapters = {
            "Persona": PersonaAdapter(),
            "Mem0": Mem0Adapter(),
            #"Zep": ZepAdapter()
        }
        self.results = []
        # Judge LLM
        self.judge_llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini"),
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
            temperature=0,
        )

    def load_questions(self, limit=None):
        questions = []
        # Priority 1: Full LongMemEval Dataset (Cleaned)
        path = "evals/data/longmemeval/longmemeval_s_cleaned.json"
        
        if os.path.exists(path):
            print(f"📖 Loading Full Dataset from {path}")
            with open(path, "r") as f:
                raw_data = json.load(f)
                
            # Convert to our internal format
            for item in raw_data:
                # Convert dates: "2023/05/20 (Sat) 02:21" -> "2023-05-20"
                # And combine haystack sessions
                sessions = []
                for i, raw_session in enumerate(item.get("haystack_sessions", [])):
                    date_str = item["haystack_dates"][i]
                    # Simple parse: split by space and take first part "2023/05/20"
                    # Then replace / with -
                    clean_date = date_str.split(" ")[0].replace("/", "-")
                    
                    # Combine messages into one block
                    content = "\n".join([f"{m['role']}: {m['content']}" for m in raw_session])
                    sessions.append({"date": clean_date, "content": content})
                
                questions.append({
                    "question": item["question"],
                    "gold": item["answer"],
                    "type": item["question_type"],
                    "sessions": sessions
                })
                
        elif os.path.exists("evals/data/longmemeval/test.json"):
             print("⚠️ Full dataset not found. Using small test set.")
             with open("evals/data/longmemeval/test.json", "r") as f:
                questions = json.load(f)

        if limit:
            questions = questions[:limit]
            
        return questions

    def evaluate_answer(self, question, gold, hypothesis) -> dict:
        prompt = f"""
        You are an impartial judge. Grade the correctness of the hypothesis answer compared to the gold answer.
        
        Question: {question}
        Gold Answer: {gold}
        Hypothesis Answer: {hypothesis}
        
        Evaluate on a scale of 1-5 where:
        1: Completely incorrect or irrelevant.
        5: Completely correct and complete.
        
        Output valid JSON: {{"grade": int, "reason": "string"}}
        """
        try:
            res = self.judge_llm.invoke(prompt)
            # basic cleanup
            content = res.content.replace("```json", "").replace("```", "")
            return json.loads(content)
        except Exception as e:
            return {"grade": 0, "reason": f"Evaluation failed: {e}"}

    def run(self, limit=2): # Start small
        questions = self.load_questions(limit)
        if not questions:
            print("No questions found! Check path.")
            return

    def run(self, limit=None):
        questions = self.load_questions(limit)
        if not questions:
            print("No questions found! Check path.")
            return

        # Check for existing checkpoint
        checkpoint_file = "evals/results/benchmark_checkpoint.jsonl"
        processed_hashes = set()
        if os.path.exists(checkpoint_file):
            print(f"🔄 Found checkpoint file: {checkpoint_file}. Resuming...")
            with open(checkpoint_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            # create a unique signature for the question to avoid duplicates
                            q_hash = f"{record['question']}_{record.get('type')}"
                            processed_hashes.add(q_hash)
                            self.results.append(record)
                        except:
                            pass
            print(f"⏩ Skipped {len(processed_hashes)} already processed questions.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_qs = len(questions)
        remaining_qs = [q for q in questions if f"{q['question']}_{q.get('type')}" not in processed_hashes]
        
        print(f"🏁 Starting Benchmark: {len(remaining_qs)} questions remaining (Total: {total_qs})")
        
        start_time = time.time()
        
        for i, q in enumerate(remaining_qs):
            loop_start = time.time()
            idx_overall = len(processed_hashes) + i + 1
            
            # ETA Calculation
            if i > 0:
                avg_time = (time.time() - start_time) / i
                eta_seconds = avg_time * (len(remaining_qs) - i)
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = "Calculating..."

            print(f"Processing ({idx_overall}/{total_qs}) [ETA: {eta_str}]: {q['question'][:50]}...", flush=True)
            user_id = f"bench_user_{idx_overall}_{timestamp}"

            row = {"question": q['question'], "gold": q['gold'], "type": q['type']}
            
            # Parallel Execution for all adapters
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def process_adapter(name, adapter, question_data):
                row_update = {}
                adapter_user_id = f"{name}_{user_id}"
                
                try:
                    # CLEAN SLATE
                    adapter.reset(adapter_user_id)
                    
                    # 1. Ingest (Bulk)
                    t_ingest_start = time.time()
                    sessions = question_data.get('sessions', [])
                    if not sessions:
                        sessions = [{"date": "2024-01-01", "content": f"Context: {question_data.get('gold', 'unknown')}"}]
                    
                    print(f"    [{name}] Ingesting {len(sessions)} sessions...", flush=True)
                    adapter.add_sessions(adapter_user_id, sessions)
                    
                    # Single Indexing Sleep per system
                    print(f"    [{name}] Indexing (5s)...", flush=True)
                    time.sleep(5)
                    
                    # 2. Query
                    t_query_start = time.time()
                    ans = adapter.query(adapter_user_id, question_data['question'])
                    duration = time.time() - t_query_start
                    
                    # 3. Eval
                    eval_res = self.evaluate_answer(question_data['question'], question_data['gold'], ans)
                    
                    row_update[f"{name}_ans"] = ans
                    row_update[f"{name}_grade"] = eval_res['grade']
                    row_update[f"{name}_reason"] = eval_res['reason']
                    row_update[f"{name}_latency"] = duration
                    
                except Exception as e:
                    print(f"  ! Error {name}: {e}")
                    row_update[f"{name}_error"] = str(e)
                    row_update[f"{name}_ans"] = "ERROR"
                    row_update[f"{name}_grade"] = 0
                
                return row_update

            
            row = {"question": q['question'], "gold": q['gold'], "type": q['type']}
            
            # Run connected systems in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(process_adapter, name, adapter, q): name for name, adapter in self.adapters.items()}
                
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        res = future.result()
                        row.update(res)
                        print(f"  > [{name}] Finished (Grade: {res.get(f'{name}_grade')})")
                    except Exception as exc:
                         print(f"  ! System {name} crashed: {exc}")
            
            # NOTE: We moved ingestion INSIDE the per-adapter loop to support unique IDs.
            # Original code had ingestion separately. This is cleaner for total isolation.
            
            self.results.append(row)
            
            # IMMEDIATE SAVE (Checkpoint)
            os.makedirs("evals/results", exist_ok=True)
            with open(checkpoint_file, "a") as f:
                f.write(json.dumps(row) + "\n")

        # Final Save (Aggregate)
        out_file = f"evals/results/benchmark_run_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Benchmark Complete! Saved to {out_file}")
        
        # Clean up checkpoint if desired, or keep it as backup. 
        # For now, we keep it.


if __name__ == "__main__":
    runner = BenchmarkRunner()
    # Run full benchmark
    runner.run()
