import json
import random
import sys

def sample_dataset(input_file, output_file, target_count=80):
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Group by type
    by_type = {}
    for q in data:
        # The key in raw dataset is 'question_type', not 'type'
        t = q.get('question_type', 'unknown')
        if t not in by_type:
             by_type[t] = []
        by_type[t].append(q)

    print(f"Dataset Counts by Type:")
    for t, qs in by_type.items():
        print(f"  {t}: {len(qs)}")

    selected_questions = []
    
    # We want to specifically target these types:
    # 1. multi-session-user (we already have plenty, but user asked for "mixed")
    # 2. temporal-reasoning
    # 3. multi-session-multi-hop? (Let's see what keys exist)
    
    # Heuristic: Grab ~40 Multi-Session and ~40 Temporal
    # We need to see the actual keys first.
    
    # Let's just grab everything that's NOT 'single-session-user' first to see what we have
    complex_types = [t for t in by_type.keys() if t != 'single-session-user']
    print(f"Complex Types Found: {complex_types}")
    
    # Stratified Sampling strategy
    # 40 questions from 'temporal' related types
    # 40 questions from 'multi-session' related types
    
    temporal_candidates = []
    multi_candidates = []
    
    for t in by_type:
        if 'temporal' in t:
            temporal_candidates.extend(by_type[t])
        elif 'multi-session' in t:
            multi_candidates.extend(by_type[t])
            
    print(f"Temporal Candidates: {len(temporal_candidates)}")
    print(f"Multi-Session Candidates: {len(multi_candidates)}")
    
    # Sample
    if temporal_candidates:
        selected_questions.extend(random.sample(temporal_candidates, min(len(temporal_candidates), 40)))
    if multi_candidates:
        selected_questions.extend(random.sample(multi_candidates, min(len(multi_candidates), 40)))
        
    print(f"\nTotal Selected: {len(selected_questions)}")
    
    with open(output_file, 'w') as f:
        json.dump(selected_questions, f, indent=2)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python sample_questions.py <input> <output>")
    else:
        sample_dataset(sys.argv[1], sys.argv[2])
