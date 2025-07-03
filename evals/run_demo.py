#!/usr/bin/env python3
"""
LongMemEval Demo Script
======================

This script demonstrates the complete LongMemEval pipeline on 3 sample instances.
It's designed to verify that the entire evaluation pipeline works end-to-end
before running on larger datasets.

Usage:
    cd evals
    python run_demo.py
"""

import asyncio
import os
import sys
from pathlib import Path
import argparse

# Add the parent directory to the path so we can import the longmemeval module
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.longmemeval.pipeline import LongMemEvalPipeline

async def run_demo(args):
    """Run a demo of the complete pipeline on 3 samples"""
    
    print("🎯 LongMemEval Demo")
    print("==================")
    print("This demo will run the complete evaluation pipeline on 3 sample instances.")
    print("It tests: download → ingest → answer → evaluate")
    print()
    
    # Check if required environment variables are set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-key-here'")
        return False
    
    # Check if the persona server is running (optional check)
    print("📡 Checking if Persona server is available...")
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/api/v1/version") as response:
                if response.status == 200:
                    print("✅ Persona server is running")
                else:
                    print("⚠️  Persona server might not be running (status: {response.status})")
    except Exception as e:
        print(f"⚠️  Could not connect to Persona server: {e}")
        print("Make sure the server is running with: python server/main.py")
    
    print()
    
    # Create and run pipeline
    pipeline = LongMemEvalPipeline(
        dataset_type="oracle",  # Use oracle dataset
        strategy="hybrid",      # Use hybrid strategy
        backend="hybrid",       # Use hybrid backend
        limit=3,                # Only 3 samples for demo
        eval_model=args.eval_model
    )
    
    result = await pipeline.run_complete_pipeline()
    
    # Print final summary
    print("\n" + "="*60)
    print("🎯 DEMO COMPLETE")
    print("="*60)
    
    if result['success']:
        print("✅ Status: SUCCESS")
        print(f"⏱️  Total time: {result['total_time']:.2f} seconds")
        print(f"📊 Accuracy: {result['evaluation_results']['overall_accuracy']:.4f}")
        print("\nNext steps:")
        print("1. Review results in evals/results/")
        print("2. Run on larger dataset: python -m evals.longmemeval.pipeline --limit 50")
        print("3. Try different strategies: --strategy vector-only")
        return True
    else:
        print("❌ Status: FAILED")
        print(f"💥 Error: {result['error']}")
        print("\nTroubleshooting:")
        print("1. Check that OPENAI_API_KEY is set")
        print("2. Ensure Persona server is running")
        print("3. Check network connectivity")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run LongMemEval demo script.")
    parser.add_argument(
        "--eval-model",
        type=str,
        default=None,
        help="Specify a model to use for evaluation (e.g., 'gpt-4.1-mini')."
    )
    args = parser.parse_args()

    print("Starting LongMemEval demo...")
    success = asyncio.run(run_demo(args))
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 