#!/usr/bin/env python3
"""
Re-evaluate benchmark results using the judge LLM.
This script loads existing results (with answers) and re-runs the evaluation step only.
"""
import json
import os
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from langchain_openai import AzureChatOpenAI
from evals.longmemeval.evaluate_qa import get_anscheck_prompt


def reevaluate(results_file: str, output_file: str):
    """Re-evaluate answers in a results JSON file."""
    
    # Judge LLM with temperature=1 for GPT-5 compatibility
    judge_llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4.1-mini"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
        temperature=1,  # CRITICAL: GPT-5/O1 requires temperature=1
    )
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Re-evaluating {len(results)} questions...")
    
    corrected = 0
    for i, r in enumerate(results):
        question = r.get('question', '')
        gold = r.get('gold', '')
        question_type = r.get('type', 'multi-session')
        
        # Find the answer key (e.g., 'Zep (Graphiti)_ans')
        ans_key = None
        for k in r.keys():
            if k.endswith('_ans'):
                ans_key = k
                break
        
        if not ans_key:
            continue
            
        hypothesis = r.get(ans_key, '')
        correct_key = ans_key.replace('_ans', '_correct')
        eval_key = ans_key.replace('_ans', '_raw_eval')
        
        # Skip if already correctly evaluated
        old_eval = r.get(eval_key, '')
        if 'Evaluation failed' not in old_eval and old_eval != '':
            continue
        
        # Generate evaluation prompt
        prompt = get_anscheck_prompt(question_type, question, gold, hypothesis, abstention=False)
        
        try:
            res = judge_llm.invoke(prompt)
            raw_response = res.content.strip().lower()
            correct = 'yes' in raw_response
            
            r[correct_key] = correct
            r[eval_key] = raw_response
            corrected += 1
            
            status = "✓" if correct else "✗"
            print(f"[{i+1}/{len(results)}] {status} Q: {question[:50]}...")
            
        except Exception as e:
            print(f"[{i+1}/{len(results)}] Error: {e}")
            r[eval_key] = f"Re-evaluation failed: {e}"
    
    # Save updated results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Re-evaluated {corrected} questions. Saved to {output_file}")
    
    # Print summary
    from collections import Counter
    types = Counter()
    correct_count = Counter()
    for r in results:
        t = r.get('type', 'unknown')
        types[t] += 1
        for k in r.keys():
            if k.endswith('_correct') and r.get(k):
                correct_count[t] += 1
                break
    
    print("\n=== Updated Summary ===")
    for t in types:
        print(f"{t}: {correct_count[t]}/{types[t]} = {correct_count[t]/types[t]*100:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input results JSON")
    parser.add_argument("--output", required=True, help="Output results JSON")
    args = parser.parse_args()
    
    reevaluate(args.input, args.output)
