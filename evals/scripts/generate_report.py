import json
import sys
import os

def generate_report(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    report_path = json_path.replace(".json", ".md")
    
    with open(report_path, 'w') as f:
        f.write("# Comparative Benchmark Report\n\n")
        
        # Summary Table
        f.write("## Summary\n")
        f.write("| System | Avg Grade | Avg Latency |\n")
        f.write("| :--- | :--- | :--- |\n")
        
        systems = ["Persona", "Mem0", "Zep"]
        for sys_name in systems:
            grades = [r.get(f"{sys_name}_grade", 0) for r in data if f"{sys_name}_grade" in r]
            latencies = [r.get(f"{sys_name}_latency", 0) for r in data if f"{sys_name}_latency" in r]
            avg_g = sum(grades)/len(grades) if grades else 0
            avg_l = sum(latencies)/len(latencies) if latencies else 0
            f.write(f"| **{sys_name}** | {avg_g:.2f} | {avg_l:.2f}s |\n")
        
        f.write("\n---\n")
        
        # Details
        f.write("## Detailed Results\n")
        for i, row in enumerate(data):
            f.write(f"### Q{i+1}: {row['question']}\n")
            f.write(f"**Type**: {row.get('type')}\n\n")
            f.write(f"**Gold Answer**: {row['gold']}\n\n")
            
            f.write("| System | Answer | Grade | Reason |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for sys_name in systems:
                ans = row.get(f"{sys_name}_ans", "N/A").replace("\n", "<br>")
                grade = row.get(f"{sys_name}_grade", "-")
                reason = row.get(f"{sys_name}_reason", "-")
                f.write(f"| **{sys_name}** | {ans} | {grade} | {reason} |\n")
            f.write("\n")

    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_report(sys.argv[1])
    else:
        # Find latest info for dev
        pass
