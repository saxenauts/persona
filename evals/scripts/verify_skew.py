import json
import numpy as np

# Load Data
def load_data(paths):
    data = []
    if isinstance(paths, str): paths = [paths]
    for p in paths:
        try:
            if p.endswith('.jsonl'):
                with open(p, 'r') as f:
                    data.extend([json.loads(line) for line in f if line.strip()])
            else:
                with open(p, 'r') as f:
                    data.extend(json.load(f))
        except Exception as e:
            print(f"Error loading {p}: {e}")
    return data

# Paths
MEM0_SINGLE = "evals/results/benchmark_run_20251215_211702.json"
ZEP_SINGLE = "evals/results/benchmark_checkpoint.jsonl"

mem0_data = load_data(MEM0_SINGLE)
zep_data = load_data(ZEP_SINGLE)

# Filter for Single Session check
mem0_single_data = [d for d in mem0_data if d.get('type') == 'single-session-user']
zep_single_data = [d for d in zep_data if d.get('type') == 'single-session-user']

# Index by Question
mem0_idx = {d['question'].strip(): d for d in mem0_single_data}

# intersection keys
zep_qs = set(d['question'].strip() for d in zep_single_data)
intersection_qs = [q for q in zep_qs if q in mem0_idx]

print(f"Total Mem0 Single Questions: {len(mem0_single_data)}")
print(f"Total Zep Single Questions: {len(zep_single_data)}")
print(f"Intersection Count: {len(intersection_qs)}")

# Calculate Stats
def calc_stats(subset_data, system_name):
    grades = []
    grade_key = f"{system_name}_grade" if f"{system_name}_grade" in subset_data[0] else None
    
    # Fallback key search
    if not grade_key:
        for k in subset_data[0].keys():
            if "_grade" in k and (system_name in k or system_name.split()[0] in k):
                grade_key = k
                break
                
    for d in subset_data:
        g = d.get(grade_key, 0)
        grades.append(g)
        
    avg = np.mean(grades)
    success = sum(1 for g in grades if g >= 4) / len(grades) * 100
    return avg, success

# 1. Mem0 Full Set (Reported Baseline)
mem0_full_avg, mem0_full_succ = calc_stats(mem0_single_data, "Mem0")

# 2. Mem0 Intersection Set (Corrected)
mem0_subset_data = [mem0_idx[q] for q in intersection_qs]
mem0_sub_avg, mem0_sub_succ = calc_stats(mem0_subset_data, "Mem0")

# 3. Zep Intersection Set
zep_sub_avg, zep_sub_succ = calc_stats(zep_single_data, "Zep")

print("\n### Correction Analysis")
print(f"**Baseline (Unfair)**:")
print(f"- Mem0 (All 70): {mem0_full_succ:.1f}% Success (Avg {mem0_full_avg:.2f})")
print(f"- Zep  (Subset): {zep_sub_succ:.1f}% Success (Avg {zep_sub_avg:.2f})")

print(f"\n**Corrected (Fair on n={len(intersection_qs)})**:")
print(f"- Mem0 (Subset): {mem0_sub_succ:.1f}% Success (Avg {mem0_sub_avg:.2f})")
print(f"- Zep  (Subset): {zep_sub_succ:.1f}% Success (Avg {zep_sub_avg:.2f})")

diff = mem0_sub_succ - mem0_full_succ
print(f"\n**Skew Impact**: Mem0 performs {abs(diff):.1f}% {'Better' if diff > 0 else 'Worse'} on this specific subset.")
