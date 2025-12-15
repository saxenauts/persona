
import json
from collections import defaultdict

def analyze_failures():
    with open("evals/results/per_question_diagnostics.json", "r") as f:
        data = json.load(f)
    
    failures = defaultdict(list)
    
    for item in data:
        if item.get("contains_gold") is False:
            q_type = item.get("question_type", "unknown")
            failures[q_type].append(item)
            
    print(f"Total failures: {sum(len(v) for v in failures.values())}")
    for q_type, items in failures.items():
        print(f"\nType: {q_type} (Count: {len(items)})")
        # Print top 3 examples
        for i, item in enumerate(items[:3]):
            print(f"--- Example {i+1} ---")
            print(f"Question: {item['question']}")
            print(f"Gold: {item['gold']}")
            print(f"Hypothesis: {item['hypothesis'][:200]}...") # Truncate hypothesis
            
if __name__ == "__main__":
    analyze_failures()
