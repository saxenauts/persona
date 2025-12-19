import json
import sys
import argparse

def analyze_results(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    full_data = data  # List of records
    
    # Auto-detect systems from keys (look for _correct suffix)
    systems = set()
    for record in full_data:
        for key in record.keys():
            if key.endswith("_correct"):
                systems.add(key.replace("_correct", ""))
    systems = sorted(list(systems)) if systems else ["Persona", "Mem0", "Zep (Graphiti)"]
    
    # Group by task type
    task_types = {}
    for record in full_data:
        t_type = record.get("type", "unknown")
        if t_type not in task_types:
            task_types[t_type] = []
        task_types[t_type].append(record)

    print(f"--- LongMemEval Binary Analysis: {len(full_data)} Questions ---")
    
    # Overall
    print(f"\n[OVERALL ACCURACY]")
    for sys_name in systems:
        correct = 0
        total = 0
        for record in full_data:
            # Support both new binary format and legacy grade format
            if f"{sys_name}_correct" in record:
                if record[f"{sys_name}_correct"]:
                    correct += 1
                total += 1
            elif f"{sys_name}_grade" in record:
                # Legacy: grade >= 4 is correct
                if record[f"{sys_name}_grade"] >= 4:
                    correct += 1
                total += 1
        
        if total > 0:
            print(f"  {sys_name}: {correct/total*100:.1f}% ({correct}/{total})")

    # Per Task Type
    print(f"\n[BREAKDOWN BY TASK TYPE]")
    for t_type, records in task_types.items():
        print(f"\nType: {t_type} ({len(records)} qs)")
        for sys_name in systems:
            correct = 0
            total = 0
            for record in records:
                if f"{sys_name}_correct" in record:
                    if record[f"{sys_name}_correct"]:
                        correct += 1
                    total += 1
                elif f"{sys_name}_grade" in record:
                    if record[f"{sys_name}_grade"] >= 4:
                        correct += 1
                    total += 1
            if total > 0:
                print(f"  {sys_name}: {correct/total*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze LongMemEval benchmark results")
    parser.add_argument("file_path", nargs="?", default="evals/results/benchmark_persona_sampled_valid.json", help="Path to the benchmark results JSON file")
    args = parser.parse_args()
    analyze_results(args.file_path)
