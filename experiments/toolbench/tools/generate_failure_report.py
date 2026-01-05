"""
Failure Analysis Report Generator
Analyzes ToolBench prediction files to determine failure reasons, decisive steps, and contributing factors.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any

STATUS_UNKNOWN = "Unknown"
STATUS_FAILED_MODEL_GIVE_UP = "Failed (Model gave up)"
STATUS_LIKELY_SUCCESS = "Likely Success"
STATUS_FAILED_ACTIVE_GIVE_UP = "Failed (Active give-up)"
STATUS_FAILED_UNKNOWN = "Failed (Unknown status)"

def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)

def analyze_task(task_id: str, data: Dict, mode: str) -> Dict:
    """
    Analyzes a single task to find failure reason and decisive step.
    """
    traj = data.get('answer_details', [])
    final_ans = data.get('final_answer', {})
    
    analysis = {
        "task_id": task_id,
        "mode": mode,
        "status": STATUS_UNKNOWN,
        "reason": STATUS_UNKNOWN,
        "decisive_step": -1,
        "factors": []
    }

    # 1. Check Success Status
    # We rely on the final answer structure or external eval (but here we infer from final answer)
    if isinstance(final_ans, dict) and final_ans.get("return_type") == "give_answer":
        # Heuristic: if final answer is short or looks like an error, it might still be a failure
        ans_text = final_ans.get("final_answer", "")
        if "sorry" in ans_text.lower() or "cannot" in ans_text.lower() or "unable" in ans_text.lower():
             analysis["status"] = STATUS_FAILED_MODEL_GIVE_UP
        else:
             analysis["status"] = STATUS_LIKELY_SUCCESS
    elif isinstance(final_ans, dict) and final_ans.get("return_type") == "give_up_and_restart":
        analysis["status"] = STATUS_FAILED_ACTIVE_GIVE_UP
    else:
        analysis["status"] = STATUS_FAILED_UNKNOWN

    # If likely success, we skip deep failure analysis unless requested
    if analysis["status"] == STATUS_LIKELY_SUCCESS:
        return analysis

    # 2. Analyze Trajectory for Decisive Failure Step
    for i, step in enumerate(traj):
        role = step.get("role")
        message = step.get("message", "")
        
        # Check for Tool Errors
        if role == "tool":
            # Check for Fake Response Errors
            if "[Fake]" in message and '"error":' in message:
                # Extract error message
                match = re.search(r'"response":\s*"(.*?)"', message)
                err_msg = match.group(1) if match else "Unknown error"
                
                if "requires" in err_msg and "parameter" in err_msg:
                    analysis["decisive_step"] = i
                    analysis["reason"] = f"Missing tool call parameters: {err_msg[:100]}..."
                    analysis["factors"].append("Model capability (empty args)")
                    analysis["factors"].append("Fake Response (explicit error)")
                    break
            
            # Check for MockExec (Cache Miss + No Fake Response) - shouldn't happen in v5 but check anyway
            if "[MockExec]" in message:
                analysis["decisive_step"] = i
                analysis["reason"] = "Cache miss with no Fake Response"
                analysis["factors"].append("Cache miss")
                break

        # Check for Model Loops or Give Up thoughts
        if role == "assistant":
            if "give up" in message.lower() or "cannot proceed" in message.lower():
                if analysis["decisive_step"] == -1:
                    analysis["decisive_step"] = i
                    analysis["reason"] = "Model decided to give up"
                    analysis["factors"].append("Model planning")

    # 3. Check for Prompt/Context Issues
    # Did the model try to use a tool that required params but didn't have them?
    # This is inferred from the "Missing Parameters" reason above.
    
    # Check if it was a Cache Miss that triggered the Fake Response
    # If we see [Fake], it implies Cache Miss.
    has_fake = any("[Fake]" in s.get("message", "") for s in traj if s.get("role") == "tool")
    if has_fake:
        analysis["factors"].append("Cache miss (triggered Fake Response)")

    return analysis

def generate_report(base_dir: str, output_file: str):
    report_lines = []
    report_lines.append("# ToolBench Failure Analysis Report")
    report_lines.append(f"**Run directory**: `{base_dir}`\n")
    report_lines.append("> **Note**: 'Likely Success' means the model produced a final answer and did not include obvious give-up keywords. Without manual verification, this is only a best-effort guess rather than a guarantee.\n")
    
    modes = ["baseline", "watermark"]
    split = "G1_instruction" # Detected or hardcoded
    
    for mode in modes:
        report_lines.append(f"## {mode.upper()} Mode")
        run_dir = os.path.join(base_dir, f"exp_v5_fake_response_{mode}", split)
        
        if not os.path.exists(run_dir):
            report_lines.append(f"*Directory not found: {run_dir}*")
            continue

        tasks = sorted([f for f in os.listdir(run_dir) if f.endswith(".json")])
        
        for task_file in tasks:
            task_id = task_file.replace(".json", "")
            data = load_json(os.path.join(run_dir, task_file))
            analysis = analyze_task(task_id, data, mode)
            
            status_icon = "[PASS]" if analysis["status"] == STATUS_LIKELY_SUCCESS else "[FAIL]"
            report_lines.append(f"### {status_icon} Task {task_id}")
            report_lines.append(f"- **Status**: {analysis['status']}")
            
            if analysis["status"] != STATUS_LIKELY_SUCCESS:
                report_lines.append(f"- **Failure reason**: {analysis['reason']}")
                report_lines.append(f"- **Decisive step**: Step {analysis['decisive_step']} (~Round {analysis['decisive_step'] // 2 + 1})")
                report_lines.append(f"- **Key factors**: {', '.join(analysis['factors'])}")
                
                # Add a brief snippet of the decisive step
                if analysis["decisive_step"] != -1:
                    traj = data.get('answer_details', [])
                    if 0 <= analysis["decisive_step"] < len(traj):
                        step_content = traj[analysis["decisive_step"]].get("message", "")
                        # Truncate for readability
                        report_lines.append(f"- **Log snippet**: `{step_content[:200]}...`")
            
            report_lines.append("")

    with open(output_file, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Report generated at {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        print("Usage: python generate_failure_report.py <run_directory>")
        sys.exit(1)
    output_file = os.path.join(base_dir, "failure_analysis_report.md")
    generate_report(base_dir, output_file)
