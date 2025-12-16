import json
import os
import glob
from collections import defaultdict

def scan_files():
    files = glob.glob("evals/results/*.json") + glob.glob("evals/results/*.jsonl")
    
    print(f"Scanning {len(files)} files...\n")
    print(f"{'Filename':<55} | {'Systems Found':<20} | {'Counts (Total/Single/Multi/Temp)'}")
    print("-" * 110)
    
    for fp in sorted(files):
        if "checkpoint" in fp and "bak" in fp: continue # Skip bak
        
        data = []
        try:
            if fp.endswith('.jsonl'):
                with open(fp, 'r') as f:
                    data = [json.loads(line) for line in f if line.strip()]
            else:
                with open(fp, 'r') as f:
                    data = json.load(f)
        except:
            continue
            
        if not data: continue
        
        # Detect Systems
        systems = set()
        first_item = data[0] if isinstance(data, list) else {}
        for k in first_item.keys():
            if k.endswith("_ans"):
                systems.add(k.replace("_ans", ""))
        
        # Count Types
        types = defaultdict(int)
        for d in data:
            t = d.get('type', 'unknown')
            types[t] += 1
            
        sys_str = ", ".join(sorted(list(systems)))
        counts_str = f"{len(data)} / {types['single-session-user']} / {types['multi-session']} / {types['temporal-reasoning']}"
        
        print(f"{os.path.basename(fp):<55} | {sys_str[:20]:<20} | {counts_str}")

if __name__ == "__main__":
    scan_files()
