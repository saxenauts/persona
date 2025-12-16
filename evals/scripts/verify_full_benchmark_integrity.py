import json

def load_qs(path, sys_name):
    qs = set()
    try:
        if path.endswith('.jsonl'):
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        qs.add(d['question'].strip())
        else:
            with open(path, 'r') as f:
                d = json.load(f)
                for item in d:
                    qs.add(item['question'].strip())
    except Exception as e:
        print(f"Error loading {sys_name}: {e}")
    return qs

# 1. Single Session Verification
print("--- Single Session Verification ---")
# Zep & Persona are in the SAME file (Checkpoint)
zep_persona_single_path = "evals/results/benchmark_checkpoint.jsonl"
mem0_single_path = "evals/results/benchmark_run_20251215_211702.json"

zp_qs = load_qs(zep_persona_single_path, "Zep/Persona (Checkpoint)")
mem0_qs = load_qs(mem0_single_path, "Mem0 (Full)")

common_single = zp_qs.intersection(mem0_qs)

print(f"Zep/Persona Count: {len(zp_qs)}")
print(f"Mem0 Count: {len(mem0_qs)}")
print(f"Fair Intersection: {len(common_single)}")

if len(common_single) == len(zp_qs):
    print("✅ Zep/Persona set is a strict subset of Mem0 set. Comparison is valid on the Intersection.")
else:
    print("❌ Mismatch in questions. Some Zep questions were NOT run by Mem0.")

# 2. Multi/Temporal Verification
print("\n--- Multi/Temporal Verification ---")
# Files
zep_multi_path = "evals/results/benchmark_run_20251216_113346.json"
persona_multi_path = "evals/results/benchmark_persona_sampled_valid.json"
mem0_multi_path = "evals/results/benchmark_mem0_vector_final.json"

zep_mqs = load_qs(zep_multi_path, "Zep Multi")
per_mqs = load_qs(persona_multi_path, "Persona Multi")
mem0_mqs = load_qs(mem0_multi_path, "Mem0 Multi")

print(f"Zep Count: {len(zep_mqs)}")
print(f"Persona Count: {len(per_mqs)}")
print(f"Mem0 Count: {len(mem0_mqs)}")

# Check Identity
if zep_mqs == per_mqs == mem0_mqs:
    print("✅ All 3 systems ran on the EXACT SAME 80 questions.")
else:
    inter = zep_mqs.intersection(per_mqs).intersection(mem0_mqs)
    print(f"❌ Mismatch. Intersection size: {len(inter)}")
    if zep_mqs != per_mqs: print(f"   Zep != Persona (Diff: {len(zep_mqs.symmetric_difference(per_mqs))})")
    if per_mqs != mem0_mqs: print(f"   Persona != Mem0 (Diff: {len(per_mqs.symmetric_difference(mem0_mqs))})")
