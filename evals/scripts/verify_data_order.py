import json

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def load_jsonl(path):
    data = []
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return data

# 1. Source Data
source = load_json('evals/data/longmemeval/single_session_user_benchmark.json')
print(f"Source Data (First 5 of {len(source)}):")
for i, item in enumerate(source[:5]):
    print(f"  {i}. {item['question']}")

# 2. Mem0 Data (Single Session?)
# Check if this file contains single session data
mem0_path = 'evals/results/benchmark_run_20251215_211702.json' 
mem0_data = load_json(mem0_path)
# Filter for single session if mixed, but assuming it might be pure single session based on user prompt
mem0_single = [d for d in mem0_data if d.get('type') == 'single-session-user']
print(f"\nMem0 Data (First 5 of {len(mem0_single)}):")
for i, item in enumerate(mem0_single[:5]):
    print(f"  {i}. {item['question']}")

# 3. Zep Data
zep_path = 'evals/results/benchmark_checkpoint.jsonl'
zep_data = load_jsonl(zep_path)
print(f"\nZep Data (First 5 of {len(zep_data)}):")
for i, item in enumerate(zep_data[:5]):
    print(f"  {i}. {item['question']}")

# Check overlap
source_qs = [q['question'] for q in source]
mem0_qs = [q['question'] for q in mem0_single]
zep_qs = [q['question'] for q in zep_data]

print("\n--- Verification ---")
if mem0_qs == source_qs[:len(mem0_qs)]:
    print(f"✅ Mem0 used the first {len(mem0_qs)} questions of the source dataset strictly in order.")
else:
    print("❌ Mem0 order mismatch or sampling detected.")

if zep_qs == source_qs[:len(zep_qs)]:
    print(f"✅ Zep used the first {len(zep_qs)} questions of the source dataset strictly in order.")
else:
    # Check if it skipped some (the resume logic skips processed ones)
    # But fundamentally, did it attempt the start?
    # Actually, verify if Zep questions are a subset of the source start?
    is_subset = all(q in source_qs[:len(zep_qs)+100] for q in zep_qs) # loose check
    print(f"⚠️ Zep Check: Strictly ordered? {zep_qs == source_qs[:len(zep_qs)]}")

