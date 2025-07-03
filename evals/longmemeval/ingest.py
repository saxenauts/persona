import asyncio
import aiohttp
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from tqdm.asyncio import tqdm as atqdm

from .config import API_BASE_URL, MAX_CONCURRENCY, RESULTS_DIR
from .loader import LongMemInstance, format_session_content, yield_instances

class PersonaIngester:
    """Handles ingestion of LongMemEval data into Persona system"""
    
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
    
    async def create_user(self, user_id: str) -> Dict:
        """Create user if not exists"""
        url = f"{self.api_base_url}/users/{user_id}"
        async with self.session.post(url) as response:
            if response.status not in [200, 201]:
                text = await response.text()
                raise Exception(f"Failed to create user {user_id}: {text}")
            return await response.json()
    
    async def ingest_session(self, user_id: str, session_content: str, session_title: str) -> Dict:
        """Ingest a session using the persona API"""
        url = f"{self.api_base_url}/users/{user_id}/ingest"
        payload = {
            "title": session_title,
            "content": session_content
        }
        async with self.session.post(url, json=payload) as response:
            if response.status != 201:
                text = await response.text()
                raise Exception(f"Failed to ingest session for {user_id}: {text}")
            return await response.json()
    
    async def process_instance(self, instance: LongMemInstance, backend_suffix: str = "hybrid") -> Dict:
        """Process a single LongMemEval instance"""
        # Create unique user ID
        user_id = f"lme_oracle_{backend_suffix}_{instance.question_id}"
        
        start_time = time.time()
        
        # Create user
        await self.create_user(user_id)
        
        # Sort sessions chronologically for temporal reasoning
        date_format = "%Y/%m/%d (%a) %H:%M"
        session_data = list(zip(instance.sessions, instance.haystack_dates))
        
        try:
            sorted_sessions = sorted(session_data, 
                                   key=lambda x: datetime.strptime(x[1], date_format))
        except ValueError as e:
            print(f"Warning: Date parsing failed for {instance.question_id}: {e}")
            # Use original order if date parsing fails
            sorted_sessions = session_data
        
        total_turns = 0
        sessions_ingested = 0
        
        for session_idx, (session_turns, session_date) in enumerate(sorted_sessions):
            if not session_turns:  # Skip empty sessions
                continue
                
            # Format session content
            session_content = format_session_content(session_turns, session_date)
            if not session_content.strip():
                continue
                
            # Create session title
            session_title = f"longmemeval_session_{session_date.replace('/', '_').replace(' ', '_').replace(':', '_')}"
            
            # Ingest session
            await self.ingest_session(user_id, session_content, session_title)
            
            sessions_ingested += 1
            total_turns += len([t for t in session_turns if t.content.strip()])
        
        processing_time = time.time() - start_time
        
        return {
            "question_id": instance.question_id,
            "user_id": user_id,
            "question_type": instance.question_type,
            "question": instance.question,
            "gold_answer": instance.gold_answer,
            "question_date": instance.question_date,
            "sessions_processed": sessions_ingested,
            "total_turns": total_turns,
            "processing_time": processing_time,
            "status": "success"
        }

async def ingest_dataset(dataset_path: str, backend: str = "hybrid", 
                        max_instances: int = None, start_idx: int = 0, 
                        end_idx: int = None) -> Dict[str, Dict]:
    """
    Ingest LongMemEval dataset into Persona system
    
    Args:
        dataset_path: Path to the dataset JSON file
        backend: Backend identifier for user ID generation
        max_instances: Maximum number of instances to process (for testing)
    
    Returns:
        Dictionary mapping question_id to ingestion results
    """
    
    # Ensure results directory exists
    results_path = Path(RESULTS_DIR)
    results_path.mkdir(parents=True, exist_ok=True)
    
    manifest = {}
    failed_instances = []
    
    print(f"🚀 Starting ingestion with {backend} backend...")
    print(f"📁 Dataset: {dataset_path}")
    if start_idx > 0 or (end_idx and end_idx != max_instances):
        print(f"🎯 Processing instances {start_idx + 1}-{end_idx or max_instances}")
    elif max_instances:
        print(f"🎯 Processing first {max_instances} instances")
    
    async with PersonaIngester() as ingester:
        # Create a list of instances to process
        instances_to_process = []
        for i, instance in enumerate(yield_instances(dataset_path)):
            if i < start_idx:
                continue
            if end_idx and i >= end_idx:
                break
            if max_instances and i >= max_instances:
                break
            instances_to_process.append(instance)

        # Process sequentially to avoid overwhelming the server
        for instance in atqdm(instances_to_process, desc="Ingesting"):
            try:
                result = await ingester.process_instance(instance, backend)
                if result["status"] == "success":
                    manifest[result["question_id"]] = result
                else:
                    failed_instances.append(result)
            except Exception as e:
                print(f"❌ Instance {instance.question_id} failed: {e}")
                failed_instances.append({
                    "question_id": instance.question_id, 
                    "status": "error", 
                    "error": str(e)
                })

    # Save manifest
    manifest_path = results_path / f"ingest_manifest_{backend}.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Print summary
    print(f"\n✅ Ingestion Complete!")
    print(f"   📊 Successful: {len(manifest)} instances")
    print(f"   ❌ Failed: {len(failed_instances)} instances")
    print(f"   📁 Manifest saved to: {manifest_path}")
    
    if failed_instances:
        failed_path = results_path / f"failed_ingestion_{backend}.json"
        with open(failed_path, 'w') as f:
            json.dump(failed_instances, f, indent=2)
        print(f"   ⚠️  Failed instances saved to: {failed_path}")
    
    # Calculate statistics
    if manifest:
        total_turns = sum(r["total_turns"] for r in manifest.values())
        total_sessions = sum(r["sessions_processed"] for r in manifest.values())
        avg_processing_time = sum(r["processing_time"] for r in manifest.values()) / len(manifest)
        
        print(f"\n📈 Statistics:")
        print(f"   Total turns ingested: {total_turns}")
        print(f"   Total sessions ingested: {total_sessions}")
        print(f"   Average processing time: {avg_processing_time:.2f}s per instance")
        print(f"   Average turns per instance: {total_turns/len(manifest):.1f}")
    
    return manifest

async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest LongMemEval data into Persona")
    parser.add_argument("dataset_path", help="Path to LongMemEval JSON file")
    parser.add_argument("--backend", choices=["vector", "hybrid"], default="hybrid", 
                       help="Memory backend type")
    parser.add_argument("--limit", type=int, help="Limit number of instances for testing")
    
    args = parser.parse_args()
    
    await ingest_dataset(args.dataset_path, args.backend, args.limit)

if __name__ == "__main__":
    asyncio.run(main()) 