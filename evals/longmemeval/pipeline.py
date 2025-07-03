#!/usr/bin/env python3
"""
LongMemEval Pipeline for Persona System
=======================================

Complete end-to-end evaluation pipeline that:
1. Downloads LongMemEval dataset from HuggingFace
2. Ingests data into Persona system
3. Generates answers using specified strategy
4. Evaluates results using official LongMemEval methodology
5. Produces comprehensive results and analysis

Usage:
    python -m evals.longmemeval.pipeline --dataset oracle --strategy hybrid --limit 3
"""

import asyncio
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import aiohttp
import os

from .config import DEFAULT_SUBSET_SIZE, RESULTS_DIR, API_BASE_URL
from .fetch_data import download_dataset
from .ingest import ingest_dataset
from .answer import generate_answers, save_results
from .evaluate_qa import evaluate_qa, print_results

class LongMemEvalPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, dataset_type: str = "oracle", strategy: str = "hybrid", 
                 backend: str = "hybrid", limit: Optional[int] = None,
                 batch_range: Optional[str] = None,
                 eval_model: Optional[str] = None):
        self.dataset_type = dataset_type
        self.strategy = strategy
        self.backend = backend
        self.limit = limit or DEFAULT_SUBSET_SIZE
        self.batch_range = batch_range
        self.eval_model = eval_model
        
        # Parse batch range if provided (use full dataset size for validation, not limit)
        dataset_size = 500 if batch_range else self.limit  # Full oracle dataset has 500 instances
        self.start_idx, self.end_idx = self._parse_batch_range(batch_range, dataset_size)
        
        # Paths
        self.results_dir = Path(RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline tracking
        self.pipeline_start_time = None
        self.stages = {}
    
    def _parse_batch_range(self, batch_range: Optional[str], limit: int) -> tuple[int, int]:
        """Parse batch range string like '1-50' or '51-100' into start and end indices.
        
        Args:
            batch_range: String like "1-50" or "51-100", or None for full range
            limit: Maximum number of instances
            
        Returns:
            Tuple of (start_idx, end_idx) where indices are 0-based for array slicing
        """
        if not batch_range:
            return 0, limit
            
        try:
            if '-' not in batch_range:
                raise ValueError("Batch range must be in format 'start-end' (e.g., '1-50')")
                
            start_str, end_str = batch_range.split('-', 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            
            if start < 1:
                raise ValueError("Start index must be 1 or greater")
            if end < start:
                raise ValueError("End index must be greater than or equal to start index")
                
            # Convert to 0-based indexing for array slicing
            start_idx = start - 1  # Convert from 1-based to 0-based
            end_idx = min(end, limit)  # Don't exceed the limit
            
            if start_idx >= limit:
                raise ValueError(f"Start index {start} exceeds dataset limit {limit}")
                
            return start_idx, end_idx
            
        except ValueError as e:
            print(f"❌ Invalid batch range '{batch_range}': {e}")
            print("   Example valid ranges: '1-50', '51-100', '101-150'")
            raise
        
    async def cleanup(self):
        """Clean up previous run artifacts for a clean slate."""
        print("\n🧹 Stage 0: Cleaning up previous run...")
        stage_start = time.time()
        
        # 1. Delete old result files
        if self.results_dir.exists():
            shutil.rmtree(self.results_dir)
            print(f"  - Deleted old results directory: {self.results_dir}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Delete users from the graph to ensure a clean state
        # We need to load the dataset to know which users to delete.
        dataset_path = self.results_dir.parent / "data" / f"longmemeval_{self.dataset_type}.json"
        if not dataset_path.exists():
            print("  - No local dataset found, skipping user deletion.")
            self.log_stage("cleanup", stage_start, time.time())
            return

        user_ids_to_delete = []
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            # Use batch range if specified, otherwise use limit
            for i, item in enumerate(data):
                if i < self.start_idx:
                    continue
                if i >= self.end_idx:
                    break
                # Construct the same user_id that will be used in ingestion
                user_id = f"lme_{self.dataset_type}_{self.backend}_{item['question_id']}"
                user_ids_to_delete.append(user_id)

        deleted_count = 0
        async with aiohttp.ClientSession() as session:
            for user_id in user_ids_to_delete:
                url = f"{API_BASE_URL}/users/{user_id}"
                async with session.delete(url) as response:
                    if response.status in [200, 404]:
                        deleted_count += 1
                    else:
                        print(f"  - Warning: Failed to delete user {user_id} (status: {response.status})")
        
        print(f"  - Attempted to delete {len(user_ids_to_delete)} users, {deleted_count} confirmed clean.")
        self.log_stage("cleanup", stage_start, time.time())

    def log_stage(self, stage_name: str, start_time: float, end_time: float, 
                  success: bool = True, details: Dict = None):
        """Log pipeline stage completion"""
        self.stages[stage_name] = {
            'duration': end_time - start_time,
            'success': success,
            'details': details or {},
            'timestamp': time.time()
        }
        
        status = "✅" if success else "❌"
        print(f"{status} {stage_name}: {end_time - start_time:.2f}s")
    
    async def run_complete_pipeline(self) -> Dict:
        """Run the complete evaluation pipeline"""
        
        self.pipeline_start_time = time.time()
        
        print("🚀 Starting LongMemEval Pipeline")
        print("="*60)
        print(f"Dataset: {self.dataset_type}")
        print(f"Strategy: {self.strategy}")
        print(f"Backend: {self.backend}")
        if self.batch_range:
            print(f"Batch Range: {self.batch_range} (instances {self.start_idx + 1}-{self.end_idx})")
        else:
            print(f"Limit: {self.limit} instances")
        print("="*60)
        
        try:
            # STAGE 0: CLEANUP
            await self.cleanup()

            # Stage 1: Download dataset
            print("\n📦 Stage 1: Downloading dataset...")
            stage_start = time.time()
            dataset_path = download_dataset(self.dataset_type)
            self.log_stage("download", stage_start, time.time(), details={'path': str(dataset_path)})
            
            # Stage 2: Ingest data
            print("\n📥 Stage 2: Ingesting data into Persona system...")
            stage_start = time.time()
            # For batch ranges, pass the end_idx as limit to avoid confusion
            effective_limit = self.end_idx if self.batch_range else self.limit
            manifest = await ingest_dataset(str(dataset_path), self.backend, effective_limit, 
                                           start_idx=self.start_idx, end_idx=self.end_idx)
            manifest_path = self.results_dir / f"ingest_manifest_{self.backend}.json"
            self.log_stage("ingest", stage_start, time.time(), 
                          details={'manifest_path': str(manifest_path), 'instances': len(manifest)})
            
            # Stage 3: Generate answers
            print("\n🤖 Stage 3: Generating answers...")
            stage_start = time.time()
            results, failed_results = await generate_answers(
                str(dataset_path), str(manifest_path), self.strategy, effective_limit,
                start_idx=self.start_idx, end_idx=self.end_idx
            )
            
            # Save results
            official_path, detailed_path = await save_results(
                results, failed_results, self.strategy, self.backend
            )
            
            self.log_stage("answer_generation", stage_start, time.time(),
                          details={'successful': len(results), 'failed': len(failed_results),
                                  'official_path': str(official_path)})
            
            # Stage 4: Evaluate results
            print("\n📊 Stage 4: Evaluating results...")
            stage_start = time.time()
            
            # Pass eval_model if it's provided
            eval_kwargs = {}
            if self.eval_model:
                eval_kwargs['metric_model'] = self.eval_model
            eval_results, eval_logs = evaluate_qa(str(official_path), str(dataset_path), **eval_kwargs)
            
            # Save evaluation results
            eval_path = self.results_dir / f"evaluation_{self.strategy}_{self.backend}.json"
            with open(eval_path, 'w') as f:
                json.dump({
                    'evaluation_results': eval_results,
                    'pipeline_stages': self.stages,
                    'configuration': {
                        'dataset_type': self.dataset_type,
                        'strategy': self.strategy,
                        'backend': self.backend,
                        'limit': self.limit
                    }
                }, f, indent=2)
            
            self.log_stage("evaluation", stage_start, time.time(),
                          details={'eval_path': str(eval_path)})
            
            # Print results
            print_results(eval_results)
            
            # Pipeline summary
            total_time = time.time() - self.pipeline_start_time
            print(f"\n🎉 Pipeline completed successfully in {total_time:.2f}s")
            print(f"📁 Results saved to: {self.results_dir}")
            
            return {
                'success': True,
                'total_time': total_time,
                'stages': self.stages,
                'evaluation_results': eval_results,
                'files': {
                    'dataset': str(dataset_path),
                    'manifest': str(manifest_path),
                    'hypotheses': str(official_path),
                    'detailed_results': str(detailed_path),
                    'evaluation': str(eval_path)
                }
            }
            
        except Exception as e:
            total_time = time.time() - self.pipeline_start_time
            print(f"\n❌ Pipeline failed after {total_time:.2f}s: {e}")
            
            return {
                'success': False,
                'total_time': total_time,
                'error': str(e),
                'stages': self.stages
            }

def print_pipeline_summary(pipeline_result: Dict):
    """Print a comprehensive pipeline summary"""
    
    print("\n" + "="*80)
    print("📊 LONGMEMEVAL PIPELINE SUMMARY")
    print("="*80)
    
    if pipeline_result['success']:
        print("✅ Status: SUCCESS")
        eval_results = pipeline_result['evaluation_results']
        print(f"📈 Overall Accuracy: {eval_results['overall_accuracy']:.4f}")
        print(f"⏱️  Total Time: {pipeline_result['total_time']:.2f}s")
        
        # Stage breakdown
        print(f"\n🔄 Stage Breakdown:")
        for stage, details in pipeline_result['stages'].items():
            status = "✅" if details['success'] else "❌"
            print(f"  {status} {stage}: {details['duration']:.2f}s")
        
        # Files generated
        print(f"\n📁 Generated Files:")
        for file_type, path in pipeline_result['files'].items():
            print(f"  {file_type}: {path}")
            
    else:
        print("❌ Status: FAILED")
        print(f"💥 Error: {pipeline_result['error']}")
        print(f"⏱️  Time to failure: {pipeline_result['total_time']:.2f}s")
    
    print("="*80)

async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LongMemEval evaluation pipeline")
    parser.add_argument("--dataset", choices=["oracle", "s", "m"], default="oracle",
                       help="Dataset type to evaluate on")
    parser.add_argument("--strategy", choices=["hybrid", "vector-only"], default="hybrid",
                       help="Answer generation strategy")
    parser.add_argument("--backend", choices=["vector", "hybrid"], default="hybrid",
                       help="Memory backend type")
    parser.add_argument("--limit", type=int, default=DEFAULT_SUBSET_SIZE,
                       help="Limit number of instances (for testing)")
    parser.add_argument("--batch", type=str, default=None,
                       help="Process specific batch range (e.g., '1-50', '51-100')")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = LongMemEvalPipeline(
        dataset_type=args.dataset,
        strategy=args.strategy,
        backend=args.backend,
        limit=args.limit,
        batch_range=args.batch
    )
    
    result = await pipeline.run_complete_pipeline()
    
    # Print summary
    print_pipeline_summary(result)
    
    # Return appropriate exit code
    return 0 if result['success'] else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main()) 