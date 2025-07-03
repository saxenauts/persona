import asyncio
import aiohttp
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional
from tqdm.asyncio import tqdm as atqdm

from .config import API_BASE_URL, EVAL_MODEL, TEMPERATURE, MAX_CONCURRENCY, RESULTS_DIR
from .loader import LongMemInstance, yield_instances

class PersonaAnswerer:
    """Handles answer generation using Persona system"""
    
    def __init__(self, api_base_url: str = API_BASE_URL):
        self.api_base_url = api_base_url
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def query_hybrid_rag(self, user_id: str, query: str) -> tuple[str, float]:
        """Query using hybrid (graph + vector) strategy"""
        url = f"{self.api_base_url}/users/{user_id}/rag/query"
        start_time = time.time()
        
        async with self.session.post(url, json={"query": query}) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"RAG query failed for {user_id}: {text}")
            result = await response.json()
        
        retrieval_time = time.time() - start_time
        return result["answer"], retrieval_time

    async def query_vector_only(self, user_id: str, query: str) -> tuple[str, float]:
        """Query using vector-only strategy"""
        url = f"{self.api_base_url}/users/{user_id}/rag/query-vector"
        start_time = time.time()
        
        async with self.session.post(url, json={"query": query}) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Vector query failed for {user_id}: {text}")
            result = await response.json()
        
        retrieval_time = time.time() - start_time
        return result["response"], retrieval_time

    async def answer_question(self, question_id: str, user_id: str, query: str, 
                            question_date: str, strategy: str = "hybrid") -> Dict:
        """Generate answer using specified strategy"""
        
        # Format query with date (following evaluation conventions)
        formatted_query = f"(date: {question_date}) {query}"
        
        start_time = time.time()
        
        try:
            if strategy == "hybrid":
                answer, retrieval_time = await self.query_hybrid_rag(user_id, formatted_query)
            elif strategy == "vector-only":
                answer, retrieval_time = await self.query_vector_only(user_id, formatted_query)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        except Exception as e:
            print(f"❌ Failed to answer question {question_id}: {e}")
            return {
                "question_id": question_id,
                "hypothesis": "",
                "strategy": strategy,
                "retrieval_time": 0,
                "total_time": 0,
                "status": "failed",
                "error": str(e)
            }
        
        total_time = time.time() - start_time
        
        return {
            "question_id": question_id,
            "hypothesis": answer,
            "strategy": strategy,
            "retrieval_time": retrieval_time,
            "total_time": total_time,
            "status": "success"
        }

async def generate_answers(dataset_path: str, manifest_path: str, 
                         strategy: str = "hybrid", max_instances: int = None,
                         start_idx: int = 0, end_idx: int = None) -> List[Dict]:
    """
    Generate answers for LongMemEval questions
    
    Args:
        dataset_path: Path to the dataset JSON file
        manifest_path: Path to the ingestion manifest
        strategy: Answer generation strategy ("hybrid" or "vector-only")
        max_instances: Maximum number of instances to process
    
    Returns:
        List of answer results
    """
    
    # Load ingestion manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"🚀 Starting answer generation with {strategy} strategy...")
    print(f"📁 Dataset: {dataset_path}")
    print(f"📋 Manifest: {manifest_path}")
    print(f"🎯 Available instances: {len(manifest)}")
    
    if start_idx > 0 or (end_idx and end_idx != max_instances):
        print(f"🔢 Processing instances {start_idx + 1}-{end_idx or max_instances}")
    elif max_instances:
        print(f"🔢 Processing first {max_instances} instances")
    
    results = []
    failed_results = []
    
    async with PersonaAnswerer() as answerer:
        tasks = []
        
        processed_count = 0
        for i, instance in enumerate(yield_instances(dataset_path)):
            if i < start_idx:
                continue
            if end_idx and i >= end_idx:
                break
            # When using batch ranges (start_idx/end_idx), don't apply max_instances limit
            # as it creates conflicts with the range logic
            if not (start_idx > 0 or end_idx) and max_instances and processed_count >= max_instances:
                break
                
            if instance.question_id not in manifest:
                print(f"⚠️  Question {instance.question_id} not found in manifest, skipping")
                continue
                
            user_id = manifest[instance.question_id]["user_id"]
            
            task = answerer.answer_question(
                instance.question_id, 
                user_id, 
                instance.question,
                instance.question_date,
                strategy
            )
            tasks.append(task)
            processed_count += 1
            
            # Process in batches
            if len(tasks) >= MAX_CONCURRENCY:
                print(f"📊 Processing batch {processed_count//MAX_CONCURRENCY}...")
                try:
                    batch_results = await atqdm.gather(*tasks, desc="Generating answers")
                    for result in batch_results:
                        if result["status"] == "success":
                            results.append(result)
                        else:
                            failed_results.append(result)
                except Exception as e:
                    print(f"❌ Batch processing failed: {e}")
                    failed_results.extend(tasks)
                
                tasks = []
        
        # Process remaining tasks
        if tasks:
            print(f"📊 Processing final batch...")
            try:
                batch_results = await atqdm.gather(*tasks, desc="Final batch")
                for result in batch_results:
                    if result["status"] == "success":
                        results.append(result)
                    else:
                        failed_results.append(result)
            except Exception as e:
                print(f"❌ Final batch processing failed: {e}")
                failed_results.extend(tasks)
    
    return results, failed_results

async def save_results(results: List[Dict], failed_results: List[Dict], 
                      strategy: str, backend: str):
    """Save results in both official and detailed formats"""
    
    # Ensure results directory exists
    results_path = Path(RESULTS_DIR)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Save official format for evaluation (required by LongMemEval scorer)
    official_path = results_path / f"hypotheses_{strategy}_{backend}.jsonl"
    with open(official_path, 'w') as f:
        for result in results:
            official_entry = {
                "question_id": result["question_id"],
                "hypothesis": result["hypothesis"]
            }
            f.write(json.dumps(official_entry) + "\n")
    
    # Save detailed results for analysis
    detailed_path = results_path / f"detailed_results_{strategy}_{backend}.json"
    with open(detailed_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save failed results if any
    if failed_results:
        failed_path = results_path / f"failed_answers_{strategy}_{backend}.json"
        with open(failed_path, 'w') as f:
            json.dump(failed_results, f, indent=2)
    
    # Print summary
    print(f"\n✅ Answer generation complete!")
    print(f"   📊 Successful: {len(results)} answers")
    print(f"   ❌ Failed: {len(failed_results)} answers")
    print(f"   📁 Official results: {official_path}")
    print(f"   📁 Detailed results: {detailed_path}")
    
    if failed_results:
        print(f"   ⚠️  Failed results: {failed_path}")
    
    # Calculate performance statistics
    if results:
        retrieval_times = [r["retrieval_time"] for r in results]
        total_times = [r["total_time"] for r in results]
        
        print(f"\n📈 Performance Statistics:")
        print(f"   Average retrieval time: {sum(retrieval_times)/len(retrieval_times):.3f}s")
        print(f"   Average total time: {sum(total_times)/len(total_times):.3f}s")
        print(f"   Min/Max retrieval time: {min(retrieval_times):.3f}s / {max(retrieval_times):.3f}s")
    
    return official_path, detailed_path

async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate answers using Persona system")
    parser.add_argument("dataset_path", help="Path to LongMemEval JSON file")
    parser.add_argument("manifest_path", help="Path to ingestion manifest")
    parser.add_argument("--strategy", choices=["hybrid", "vector-only"], default="hybrid", 
                       help="Answer generation strategy")
    parser.add_argument("--backend", choices=["vector", "hybrid"], default="hybrid", 
                       help="Backend type (for naming)")
    parser.add_argument("--limit", type=int, help="Limit number of instances for testing")
    
    args = parser.parse_args()
    
    results, failed_results = await generate_answers(
        args.dataset_path, 
        args.manifest_path, 
        args.strategy, 
        args.limit
    )
    
    await save_results(results, failed_results, args.strategy, args.backend)

if __name__ == "__main__":
    asyncio.run(main()) 