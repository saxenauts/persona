import json
import os
import glob
import numpy as np

# Paths
PERSONA_PATH = "evals/results/benchmark_persona_sampled_valid.json" # Multi
PERSONA_SINGLE_PATH = "evals/results/benchmark_zep_single_full.json" # Single (Persona run here)

MEM0_PATH = "evals/results/benchmark_mem0_vector_final.json"
MEM0_SINGLE_PATH = "evals/results/benchmark_run_20251215_211702.json"

ZEP_PATH = "evals/results/benchmark_run_20251216_113346.json" # Multi
ZEP_SINGLE_PATH = "evals/results/benchmark_checkpoint.jsonl" # Single

LOG_DIR = "evals/results/zep_stage_logs"

def load_data(paths):
    data = []
    if isinstance(paths, str): paths = [paths]
    for p in paths:
        if not os.path.exists(p): continue
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

persona_data = load_data([PERSONA_PATH, PERSONA_SINGLE_PATH])
mem0_data = load_data([MEM0_PATH, MEM0_SINGLE_PATH])
zep_data = load_data([ZEP_PATH, ZEP_SINGLE_PATH])

# Index by Question
def index_by_q(data, system):
    idx = {}
    for d in data:
        q = d['question'].strip()
        idx[q] = d
    return idx

persona_idx = index_by_q(persona_data, "Persona")
mem0_idx = index_by_q(mem0_data, "Mem0")
zep_idx = index_by_q(zep_data, "Zep")

# Overlap Analysis
common_qs = set(zep_idx.keys())
print(f"Total Zep Questions: {len(common_qs)}")

# Latency
latencies = {"Persona": [], "Mem0": [], "Zep (Graphiti)": []}

for q in common_qs:
    if q in persona_idx:
        l = persona_idx[q].get('Persona_latency', 0)
        if l > 0: latencies["Persona"].append(l)
        
    if q in mem0_idx:
        # Mem0 keys varied "Mem0 (Vector)_latency" or "Mem0_latency"
        l = mem0_idx[q].get('Mem0 (Vector)_latency') or mem0_idx[q].get('Mem0_latency') or 0
        if l > 0: latencies["Mem0"].append(l)
        
    if q in zep_idx:
        l = zep_idx[q].get('Zep (Graphiti)_latency', 0)
        if l > 0: latencies["Zep (Graphiti)"].append(l)

print("### Speed Analysis (Latency per Query)")
for sys, lats in latencies.items():
    if lats:
        print(f"- **{sys}**: {np.mean(lats):.2f}s (Min: {np.min(lats):.2f}, Max: {np.max(lats):.2f})")

# Interesting Cases
print("\n### Deep Dive Examples")

cases = []

# Case 1: Zep Win (Zep >= 4, Others <= 3)
for q in common_qs:
    z_res = zep_idx[q]
    z_grade = z_res.get('Zep (Graphiti)_grade', 0)
    
    p_res = persona_idx.get(q, {})
    p_grade = p_res.get('Persona_grade', 5) # Assume high if missing to be strict
    
    m_res = mem0_idx.get(q, {})
    m_grade = m_res.get('Mem0 (Vector)_grade') or m_res.get('Mem0_grade') or 5
    
    if z_grade >= 4 and p_grade <= 3 and m_grade <= 3:
        cases.append(("ZEP_WIN", q, z_res, p_res, m_res))
        if len(cases) >= 2: break

# Case 2: Zep Fail (Zep <= 2)
for q in common_qs:
    z_res = zep_idx[q]
    z_grade = z_res.get('Zep (Graphiti)_grade', 0)
    if z_grade <= 2:
        cases.append(("ZEP_FAIL", q, z_res, persona_idx.get(q,{}), mem0_idx.get(q,{})))
        break

# Extract Context from Logs for Cases
def get_log_context(q_text, zep_ans):
    # Log files are named Zep (Graphiti)_q{ID}_{UUID}.jsonl
    # We don't verify ID easily, but we can grep the question in the log file
    # Or just grep the answer/question overlap logic
    
    # Brute force search in latest logs
    logs = glob.glob(os.path.join(LOG_DIR, "*.jsonl"))
    # Sort by time desc
    logs.sort(key=os.path.getmtime, reverse=True)
    
    for l_path in logs[:50]: # Search recent
        try:
            with open(l_path, 'r') as f:
                lines = f.readlines()
                # Check generation stage
                for line in lines:
                    if "stage4_generation" in line:
                        data = json.loads(line)
                        if data.get('query') == q_text:
                            # Start looking for retrieval stage in same file
                            retrieval = None
                            for l2 in lines:
                                if "stage3_retrieval" in l2:
                                    retrieval = json.loads(l2)
                                    break
                            return retrieval
        except: pass
    return None

for c_type, q, z, p, m in cases:
    print(f"\n#### [{c_type}] {q}")
    print(f"**Gold:** {z['gold']}")
    print(f"**Zep ({z.get('Zep (Graphiti)_grade')}):** {z.get('Zep (Graphiti)_ans')}")
    print(f"**Persona ({p.get('Persona_grade')}):** {p.get('Persona_ans')}")
    print(f"**Mem0 ({m.get('Mem0 (Vector)_grade') or m.get('Mem0_grade')}):** {m.get('Mem0 (Vector)_ans') or m.get('Mem0_ans')}")
    
    ctx = get_log_context(q, z.get('Zep (Graphiti)_ans'))
    if ctx:
        print(f"**Zep Stats:** Edges: {ctx.get('edges_count')}, Nodes: {ctx.get('nodes_count')}")
        print(f"**Zep Context Preview:**\n{ctx.get('context_preview')}")
    else:
        print(f"**Zep Stats:** Log not found.")
