import json
import os
import glob

# Map systems to ALL data files containing their results
files_map = {
    "Persona": [
        "evals/results/benchmark_persona_sampled_valid.json",
        "evals/results/benchmark_checkpoint.jsonl"  # partial new run
    ],
    "Mem0": [
        "evals/results/benchmark_mem0_vector_final.json",
        "evals/results/benchmark_run_20251215_211702.json"
    ],
    "Zep (Graphiti)": [
        "evals/results/benchmark_run_20251216_113346.json",
        "evals/results/benchmark_checkpoint.jsonl"
    ]
}

def analyze_system(name, filepaths):
    merged_data = []
    for fp in filepaths:
        if os.path.exists(fp):
            try:
                if fp.endswith('.jsonl'):
                    with open(fp, 'r') as f:
                         chunk = [json.loads(line) for line in f if line.strip()]
                         merged_data.extend(chunk)
                else:
                    with open(fp, 'r') as f:
                        chunk = json.load(f)
                        merged_data.extend(chunk)
            except:
                print(f"⚠️ Failed to load {fp}")
                    
    if not merged_data:
        print(f"⚠️ No data found for {name}")
        return None
        
    stats = {}
    
    # Identify question types
    types = set(d.get('type', 'unknown') for d in merged_data)
    
    # Determine grade key dynamically per chunk? 
    # Better to just look for the key in the merged data
    # (assuming all chunks use same key format for the system)
    grade_key = None
    for d in merged_data:
        for k in d.keys():
            if k.endswith("_grade") and (name in k or name.split()[0] in k):
                grade_key = k
                break
        if grade_key: break
            
    if not grade_key:
         if name == "Persona": grade_key = "Persona_grade"
         elif name == "Mem0": grade_key = "Mem0 (Vector)_grade"
         elif name == "Zep (Graphiti)": grade_key = "Zep (Graphiti)_grade"

    for q_type in types:
        category_data = [d for d in merged_data if d.get('type') == q_type]
        # Filter for data that actually has the system's grade (some files might lack it)
        category_data = [d for d in category_data if grade_key in d or d.get(grade_key) is not None]
        
        count = len(category_data)
        if count == 0: continue
            
        grades = [d.get(grade_key, 0) for d in category_data]
        avg_grade = sum(grades) / count
        success_count = sum(1 for g in grades if g >= 4)
        success_rate = (success_count / count) * 100
        
        stats[q_type] = {
            "count": count,
            "avg_grade": avg_grade,
            "success_rate": success_rate,
            "system_grade_key": grade_key
        }
        
    return stats, merged_data

print("### Detailed Breakdown by Question Type\n")
print("| System | Type | Count | Avg Grade | Success Rate (>=4) |")
print("|---|---|---|---|---|")

overall_stats = {}

for name, filepaths in files_map.items():
    res = analyze_system(name, filepaths)
    if res:
        stats, full_data = res
        grade_key = None
        
        for q_type, metrics in stats.items():
            print(f"| **{name}** | {q_type} | {metrics['count']} | {metrics['avg_grade']:.2f} | {metrics['success_rate']:.1f}% |")
            grade_key = metrics['system_grade_key']
            
        # Calc Overall
        all_relevant = [d for d in full_data if grade_key in d or d.get(grade_key) is not None]
        if all_relevant:
            grades = [d.get(grade_key, 0) for d in all_relevant]
            avg = sum(grades) / len(grades)
            succ = sum(1 for g in grades if g >= 4)
            rate = (succ / len(grades)) * 100
            overall_stats[name] = (avg, rate, len(grades))

print("\n\n### Overall Summary\n")
print("| System | Count | Avg Grade | Success Rate |")
print("|---|---|---|---|")
for name, (avg, rate, count) in overall_stats.items():
    print(f"| {name} | {count} | {avg:.2f} | {rate:.1f}% |")
