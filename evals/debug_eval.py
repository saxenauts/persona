#!/usr/bin/env python3
"""
Debug script to test evaluation stage in pipeline context
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evals.longmemeval.evaluate_qa import evaluate_qa, print_results

def main():
    print("🔍 Debug: Testing evaluation stage in pipeline context")
    
    # Test with the exact same parameters as the pipeline
    hypotheses_file = "evals/results/hypotheses_hybrid_hybrid.jsonl"
    reference_file = "evals/data/longmemeval_oracle.json"
    
    print(f"📁 Hypotheses file: {hypotheses_file}")
    print(f"📁 Reference file: {reference_file}")
    print(f"🔑 API Key set: {'OPENAI_API_KEY' in os.environ}")
    print(f"📍 Working directory: {os.getcwd()}")
    
    # Check if files exist
    if not os.path.exists(hypotheses_file):
        print(f"❌ Hypotheses file not found: {hypotheses_file}")
        return
    
    if not os.path.exists(reference_file):
        print(f"❌ Reference file not found: {reference_file}")
        return
    
    try:
        print("\n🚀 Starting evaluation...")
        results, logs = evaluate_qa(hypotheses_file, reference_file, metric_model="gpt-4o-mini")
        print("✅ Evaluation completed successfully!")
        print_results(results)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 