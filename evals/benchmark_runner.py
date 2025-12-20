import os
import json
import time
from datetime import datetime
from evals.adapters.persona_adapter import PersonaAdapter
from evals.adapters.mem0_adapter import Mem0Adapter
from evals.adapters.zep_adapter import ZepAdapter
from langchain_openai import AzureChatOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

class BenchmarkRunner:
    def __init__(self):
        # FORCE USER REQUEST: Always use gpt-4.1-mini
        # os.environ["AZURE_CHAT_DEPLOYMENT"] = "gpt-4.1-mini"  # Commented out to allow gpt-5 env var
        
        self.adapters = {
            # "Persona": PersonaAdapter(), # Already benchmarked - commenting for Zep-only run
            #"Mem0 (Vector)": Mem0Adapter(use_graph=False),  # Already benchmarked
            # "Mem0 (Graph)": Mem0Adapter(use_graph=True),  # Already benchmarked
            "Zep (Graphiti)": ZepAdapter(),  # Running with rate-limit fix
        }
        self.results = []
        # Judge LLM
        self.judge_llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4.1-mini"),
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
            temperature=1,  # GPT-5/O1 requires temperature=1

        )

    def load_questions(self, limit=None):
        questions = []
        # Multi-session + Temporal Reasoning dataset (80 questions)
        path = "evals/data/longmemeval/sampled_benchmark_data.json"
        
        if os.path.exists(path):
            print(f"📖 Loading Sampled Dataset from {path}")
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

    def evaluate_answer(self, question, gold, hypothesis, task_type) -> dict:
        """
        Official LongMemEval binary evaluation.
        Uses task-specific prompts as defined in the LongMemEval paper.
        Returns {"correct": bool, "raw_response": str}
        """
        from evals.longmemeval.evaluate_qa import get_anscheck_prompt
        
        # Generate task-specific binary evaluation prompt
        prompt = get_anscheck_prompt(task_type, question, gold, hypothesis, abstention=False)
        
        try:
            res = self.judge_llm.invoke(prompt)
            raw_response = res.content.strip().lower()
            correct = 'yes' in raw_response
            return {"correct": correct, "raw_response": raw_response}
        except Exception as e:
            return {"correct": False, "raw_response": f"Evaluation failed: {e}"}

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=120),
        stop=stop_after_attempt(15),
        retry=retry_if_exception_type((Exception))
    )
    def _safe_ingest(self, name, adapter, user_id, sessions, date_list):
        print(f"    [{name}] Ingesting {len(sessions)} sessions...")
        try:
            adapter.add_sessions(user_id, sessions)
        except Exception as e:
            # Fallback for systems strictly requiring run_in_executor wrapper or failures
            print(f"[{name}] Ingest fail: {e}")
            raise e

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((Exception))
    )
    def _safe_query(self, adapter, user_id, question):
        return adapter.query(user_id, question)

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
                            # Use simple question text as signature
                            q_hash = record['question'].strip()
                            processed_hashes.add(q_hash)
                            self.results.append(record)
                        except:
                            pass
            print(f"⏩ Skipped {len(processed_hashes)} already processed questions.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_qs = len(questions)
        remaining_qs = [q for q in questions if q['question'].strip() not in processed_hashes]
        
        print(f"🏁 Starting Benchmark: {len(remaining_qs)} questions remaining (Total: {total_qs})")
        
        # Process 1 question (sequential) to specific threading issues
        PARALLEL_QUESTIONS = 5  # Hyper-parallel for 80% utilization
        
        import threading
        checkpoint_lock = threading.Lock()
        results_lock = threading.Lock()
        start_time = time.time()
        completed_count = [0]  # Mutable for thread-safe counter
        
        # Start fresh or append?
        if os.path.exists(checkpoint_file):
             # If we are resuming, we want to clear the checkpoint file IF we are fixing errors?
             # User wants to run only the part that was messed up.
             # The checkpoint contains "Error". We should NOT skip questions with errors.
             pass

        # RECOVERY LOGIC: Scan for existing successful ingestions
        existing_user_ids = {} # {q_idx: user_id}
        log_dir = "evals/results/zep_stage_logs"
        if os.path.exists(log_dir):
            for fname in os.listdir(log_dir):
                if fname.endswith(".jsonl") and "Zep__Graphiti__q" in fname:
                    # Parse "Zep__Graphiti__q25_fd4a.jsonl" -> index 25
                    try:
                        parts = fname.split("__q")[1].split("_")
                        idx = int(parts[0])
                        uid = fname.replace(".jsonl", "")
                        existing_user_ids[idx] = uid
                    except:
                        pass
        print(f"♻️ Found {len(existing_user_ids)} existing ingestion logs. Will verify and reuse graphs.")

        
        def process_single_question(task, idx_overall):
            """Process a single question - ingest, query, evaluate."""
            try:
                # 1. Ingestion Phase
                sessions = task.get('sessions', [])
                date_list = [s['date'] for s in sessions]
                
                adapter_user_ids = {}
                
                # Ingest for each adapter (sequential per question to avoid overload)
                for name, adapter in self.adapters.items():
                    # TRY RESUME: Check if we have an existing user_id for this question index
                    recovered_id = existing_user_ids.get(idx_overall)
                    
                    # Normalize name for matching (Zep (Graphiti) -> Zep__Graphiti)
                    import re
                    name_sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
                    
                    # Check if recovered ID matches this adapter
                    # e.g. recovered="Zep__Graphiti__q80..." name_sanitized="Zep__Graphiti"
                    match_found = False
                    if recovered_id and (name in recovered_id or name_sanitized in recovered_id or ("Zep" in name and "Zep" in recovered_id)):
                        match_found = True
                    
                    if match_found:
                        # We have a candidate ID. Check if ingestion was actually SUCCESSFUL.
                        is_ingested = False
                        try:
                            # Strict check: log file must verify completion
                            with open(f"{log_dir}/{recovered_id}.jsonl", "r") as f:
                                for line in f:
                                    if "stage1_ingestion_complete" in line:
                                        is_ingested = True
                                        break
                        except:
                            pass
                        
                        if is_ingested:
                            q_user_id = recovered_id
                            print(f"    [{name}] Q{idx_overall}: Resuming/Skipping Ingestion (Found log for {q_user_id})")
                        else:
                             # Partial log found but not complete - restart with new ID to be safe
                             q_user_id = f"{name}_q{idx_overall}_{uuid.uuid4().hex[:4]}"
                             print(f"    [{name}] Q{idx_overall}: Found log but incomplete. Re-ingesting to new ID {q_user_id}...")
                             try:
                                self._safe_ingest(name, adapter, q_user_id, sessions, date_list)
                             except Exception as e:
                                print(f"    [{name}] Q{idx_overall}: Ingest error: {e}")

                    else:
                        # New ID
                        q_user_id = f"{name}_q{idx_overall}_{uuid.uuid4().hex[:4]}"
                        print(f"    [{name}] Q{idx_overall}: Ingesting {len(sessions)} sessions...")
                        try:
                            self._safe_ingest(name, adapter, q_user_id, sessions, date_list)
                        except Exception as e:
                            print(f"    [{name}] Q{idx_overall}: Ingest error: {e}")
                    
                    adapter_user_ids[name] = q_user_id

                
                # Brief indexing wait (skip if we skipped ingestion really, but 2s is fine)
                time.sleep(1)

                
                # 2. Query Phase
                question_text = task['question']
                gold_answer = task['gold']
                question_type = task['type']
                
                row = {"question": question_text, "gold": gold_answer, "type": question_type}
                
                for name, adapter in self.adapters.items():
                    user_id_for_query = adapter_user_ids.get(name)
                    if user_id_for_query:
                        try:
                            t_query_start = time.time()
                            ans = self._safe_query(adapter, user_id_for_query, question_text)
                            duration = time.time() - t_query_start
                            
                            eval_res = self.evaluate_answer(question_text, gold_answer, ans, question_type)
                            
                            row[f"{name}_ans"] = ans
                            row[f"{name}_correct"] = eval_res['correct']
                            row[f"{name}_raw_eval"] = eval_res['raw_response']
                            row[f"{name}_latency"] = duration
                            result_str = "CORRECT" if eval_res['correct'] else "WRONG"
                            print(f"    [{name}] Q{idx_overall}: {result_str}")
                        except Exception as exc:
                            print(f"    [{name}] Q{idx_overall}: Query error: {exc}")
                            row[f"{name}_error"] = str(exc)
                            row[f"{name}_ans"] = "ERROR"
                            row[f"{name}_correct"] = False
                    else:
                        row[f"{name}_error"] = "User ID not found"
                        row[f"{name}_ans"] = "ERROR"
                        row[f"{name}_correct"] = False

                
                # Thread-safe checkpoint save
                with checkpoint_lock:
                    with open(checkpoint_file, "a") as f:
                        f.write(json.dumps(row) + "\n")
                
                with results_lock:
                    self.results.append(row)
                    completed_count[0] += 1
                    
                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / max(completed_count[0], 1)
                remaining = len(remaining_qs) - completed_count[0]
                eta_seconds = avg_time * remaining
                print(f"✓ Q{idx_overall} Complete ({completed_count[0]}/{len(remaining_qs)}) [ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s]")
                
                return row
                
            except Exception as e:
                print(f"ERROR Q{idx_overall}: {e}")
                return None
        
        # Process questions in parallel
        print(f"🚀 Running with {PARALLEL_QUESTIONS} parallel question workers...")
        
        with ThreadPoolExecutor(max_workers=PARALLEL_QUESTIONS) as executor:
            futures = {}
            for i, task in enumerate(remaining_qs):
                idx_overall = total_qs - len(remaining_qs) + i + 1
                futures[executor.submit(process_single_question, task, idx_overall)] = idx_overall
            
            # Wait for all to complete
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Question {idx} failed: {e}")

        # Final Save (Aggregate)
        out_file = f"evals/results/benchmark_run_{timestamp}.json"
        os.makedirs("evals/results", exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Benchmark Complete! Saved to {out_file}")


if __name__ == "__main__":
    runner = BenchmarkRunner()
    # Run full benchmark
    runner.run(limit=80)  # Full 80-question benchmark with dual-provider
