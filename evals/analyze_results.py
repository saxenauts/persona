import json
import sys
import numpy as np
import argparse

def analyze_results(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Convert list of dicts to dict of lists for easier analysis
    full_data = data # It's a list of records
    
    systems = ["Persona", "Mem0"]
    # Group by task type
    task_types = {}
    
    for record in full_data:
        t_type = record.get("type", "unknown")
        if t_type not in task_types:
             task_types[t_type] = []
        task_types[t_type].append(record)

    print(f"--- Analysis of {len(full_data)} Questions ---")
    
    # Overall
    print(f"\n[OVERALL]")
    for sys_name in systems:
        correct = 0
        total = 0
        grades = []
        for record in full_data:
             if f"{sys_name}_grade" in record:
                 grade = record[f"{sys_name}_grade"]
                 grades.append(grade)
                 if grade >= 4:
                     correct += 1
                 total += 1
        
        if total > 0:
            print(f"  {sys_name}: {correct/total*100:.1f}% ({correct}/{total}) | Avg: {np.mean(grades):.2f}")

    # Per Task Type
    print(f"\n[BREAKDOWN BY TASK TYPE]")
    for t_type, records in task_types.items():
        print(f"\nType: {t_type} ({len(records)} qs)")
        for sys_name in systems:
            correct = 0
            total = 0
            grades = []
            for record in records:
                 if f"{sys_name}_grade" in record:
                     grade = record[f"{sys_name}_grade"]
                     grades.append(grade)
                     if grade >= 4:
                         correct += 1
                     total += 1
            if total > 0:
                print(f"  {sys_name}: {correct/total*100:.1f}% | Avg: {np.mean(grades):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("file_path", nargs="?", default="evals/results/benchmark_persona_sampled_valid.json", help="Path to the benchmark results JSON file")
    args = parser.parse_args()
    analyze_results(args.file_path)
