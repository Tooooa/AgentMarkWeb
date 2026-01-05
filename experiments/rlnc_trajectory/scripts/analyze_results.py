import json
import os
import glob
import numpy as np
import re
from pathlib import Path

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def analyze_results(base_dir):
    # Find experiment directories
    # Assuming structure: base_dir / exp_name / test_instruction / split / query_id.json
    
    # We need to find the baseline and watermark directories
    # Based on previous steps, they are likely:
    # exp_v2_G1_category_20_baseline
    # exp_v2_G1_category_20_watermark
    
    baseline_dir = None
    watermark_dir = None
    
    for item in os.listdir(base_dir):
        if "baseline" in item and os.path.isdir(os.path.join(base_dir, item)):
            baseline_dir = os.path.join(base_dir, item)
        if "watermark" in item and os.path.isdir(os.path.join(base_dir, item)):
            watermark_dir = os.path.join(base_dir, item)
            
    if not baseline_dir or not watermark_dir:
        print("Error: Could not find baseline or watermark directories in", base_dir)
        return

    print(f"Baseline Dir: {baseline_dir}")
    print(f"Watermark Dir: {watermark_dir}")

    # Load Pass Rate Data
    eval_dir = os.path.join(base_dir, "eval")
    baseline_pass_rate = {}
    watermark_pass_rate = {}
    
    if os.path.exists(eval_dir):
        for f in os.listdir(eval_dir):
            if "baseline" in f and f.endswith(".json"):
                data = load_json(os.path.join(eval_dir, f))
                pass
    
    # Let's rely on the prediction files themselves if they contain "is_solved" (some versions do)
    # Or just count "AnswerStatus.Solved" if we can find the eval json.
    
    # Helper to get stats from a run directory
    def get_stats(run_dir):
        stats = {
            "steps": [],
            "ids": [],
            "bits": [],
            "success": [] # We might not have this directly without eval file
        }
        
        # Walk through all json files
        json_files = glob.glob(os.path.join(run_dir, "**", "*.json"), recursive=True)
        
        for jf in json_files:
            try:
                data = load_json(jf)
                if "total_steps" in data:
                    stats["steps"].append(data["total_steps"])
                    stats["ids"].append(Path(jf).stem)
                
                if "watermark_trace" in data and data["watermark_trace"]:
                    # Calculate bits - heuristic fallback
                    # Prefer using decode summary if available later, but for now:
                    # Sum diffs of valid steps
                    trace = data["watermark_trace"]
                    count = 0
                    if trace:
                        # Try to sum per-step increments 
                        # (more robust than end-start if gaps exist or intermediate indices missing)
                        for t in trace:
                            start = t.get("bit_index_before")
                            end = t.get("bit_index_after")
                            if start is not None and end is not None:
                                count += max(0, end - start)
                            elif start is not None:
                                # Fallback: if start exists but end missing, assume 1 bit embedded (common in RLNC loose extraction)
                                count += 1
                    
                    # Improved heuristic: sum of per-step width
                    stats["bits"].append(count)
                elif "watermark" in run_dir:
                     stats["bits"].append(0)
                     
            except Exception as e:
                print(f"Error reading {jf}: {e}")
        
        return stats
        
    # Process Score Calculation Logic
    def calculate_process_score(steps):
        succeed_tool_calling = 0
        used_tool_types = set()
        for step in steps:
            if step['name'] == 'Finish':
                continue
            response = str(step.get('response', ''))
            # Heuristic for success: no explicit error message in response or explicitly empty error
            if '"error": ""' in response or "'error': ''" in response:
                succeed_tool_calling += 1
            elif "[Fake]" in response or "[FakeCache]" in response: 
                # If fake/fakecache but has explicit error not empty, fail. 
                # Otherwise assume success for fake
                if '"error": "' in response and '"error": ""' not in response:
                    pass 
                else:
                    succeed_tool_calling += 1
            used_tool_types.add(step['name'])
            
        total_steps = len(steps)
        if total_steps == 0:
            return 0
            
        score = succeed_tool_calling * 10 + len(used_tool_types) * 5 - 5 * np.log(total_steps)
        return score

    def get_process_stats(run_dir):
        scores = []
        json_files = glob.glob(os.path.join(run_dir, "**", "*.json"), recursive=True)
        for jf in json_files:
            try:
                data = load_json(jf)
                # Reconstruct step list from structured format if available. 
                steps = []
                if "answer_details" in data:
                    raw_details = data["answer_details"]
                    for d in raw_details:
                        if d.get("role") == "tool":
                             msg = d.get("message", "")
                             # Parse "[(Fake)Cache/Fake/MockExec] tool=... category=... api=... args=... | response=..."
                             match = re.search(r"tool=(.*?) category=.*? response=(.*)", msg, re.DOTALL)
                             if match:
                                 tool_name = match.group(1).split(" ")[0] 
                                 resp = match.group(2)
                                 steps.append({'name': tool_name, 'response': resp})
                        elif d.get("role") == "assistant" and "Final Answer" in d.get("message", ""):
                             pass
                    
                elif "answer_steps" in data:
                    pass
                
                if steps:
                     scores.append(calculate_process_score(steps))
                     
            except Exception:
                pass
        return scores

    baseline_process_scores = get_process_stats(baseline_dir)
    watermark_process_scores = get_process_stats(watermark_dir)

    baseline_stats = get_stats(baseline_dir)
    watermark_stats = get_stats(watermark_dir)

    def get_cache_stats(run_dir):
        total_calls = 0
        real_hits = 0
        fake_new = 0
        fake_hits = 0
        mock_exec = 0
        
        json_files = glob.glob(os.path.join(run_dir, "**", "*.json"), recursive=True)
        
        for jf in json_files:
            try:
                data = load_json(jf)
                if "answer_details" in data:
                    for step in data["answer_details"]:
                        if step.get("role") == "tool":
                            msg = step.get("message", "")
                            if msg.startswith("[Cache]"):
                                real_hits += 1
                                total_calls += 1
                            elif msg.startswith("[FakeCache]"):
                                fake_hits += 1
                                total_calls += 1
                            elif msg.startswith("[Fake]"):
                                fake_new += 1
                                total_calls += 1
                            elif msg.startswith("[MockExec]"):
                                mock_exec += 1
                                total_calls += 1
            except Exception:
                pass
                
        return {
            "total": total_calls,
            "real_hits": real_hits,
            "fake_hits": fake_hits,
            "fake_new": fake_new,
            "real_rate": (real_hits / total_calls * 100) if total_calls > 0 else 0,
            "fake_hit_rate": (fake_hits / (fake_hits + fake_new) * 100) if (fake_hits + fake_new) > 0 else 0
        }
    
    baseline_cache = get_cache_stats(baseline_dir)
    watermark_cache = get_cache_stats(watermark_dir)

    # Load Eval Results
    def load_pass_rate(name):
        # Find file matching *_{name}.json in eval_dir
        pattern = os.path.join(eval_dir, f"*_{name}.json")
        files = glob.glob(pattern)
        if not files:
            return 0
            
        path = files[0]
        solved_count = 0
        total_count = 0
        
        try:
            data = load_json(path)
            if isinstance(data, dict):
                total_count = len(data)
                for qid, info in data.items():
                    if isinstance(info, dict) and "is_solved" in info:
                        is_solved = info["is_solved"]
                        if isinstance(is_solved, dict):
                            if any(v == "AnswerStatus.Solved" for v in is_solved.values()):
                                solved_count += 1
                        elif is_solved == "AnswerStatus.Solved":
                            solved_count += 1
                    elif isinstance(info, str) and info == "AnswerStatus.Solved":
                        solved_count += 1
        except Exception as e:
            print(f"Error parsing eval file {path}: {e}")
        
        if total_count > 0:
            return (solved_count / total_count) * 100
        return 0

    baseline_pr = load_pass_rate("baseline")
    watermark_pr = load_pass_rate("watermark")

    # Time Analysis (Estimate based on file mtimes)
    def get_avg_time(run_dir):
        json_files = glob.glob(os.path.join(run_dir, "**", "*.json"), recursive=True)
        if not json_files:
            return 0
        
        total_duration = 0
        count = 0
        for jf in json_files:
            try:
                data = load_json(jf)
                if "duration" in data:
                    total_duration += data["duration"]
                    count += 1
            except:
                pass
        
        if count > 0:
            return total_duration / count
            
        json_files.sort(key=os.path.getmtime)
        if len(json_files) < 2:
            return 0
        
        start_time = os.path.getmtime(json_files[0])
        end_time = os.path.getmtime(json_files[-1])
        return (end_time - start_time) / len(json_files)

    baseline_time = get_avg_time(baseline_dir)
    watermark_time = get_avg_time(watermark_dir)

    report_lines = []
    report_lines.append("="*40)
    report_lines.append("Analysis Summary")
    report_lines.append("="*40)
    
    report_lines.append(f"\n1. Success Rate Comparison:")
    report_lines.append(f"   - Baseline: {baseline_pr:.2f}%")
    report_lines.append(f"   - Watermark: {watermark_pr:.2f}%")
    
    report_lines.append(f"\n2. Steps Comparison:")
    report_lines.append(f"   - Baseline: Mean={np.mean(baseline_stats['steps']):.2f}, Max={np.max(baseline_stats['steps'])}")
    report_lines.append(f"   - Watermark: Mean={np.mean(watermark_stats['steps']):.2f}, Max={np.max(watermark_stats['steps'])}")
    
    report_lines.append(f"\n3. Time Comparison (Estimated Avg Time per Task):")
    report_lines.append(f"   - Baseline: ~{baseline_time:.2f} s")
    report_lines.append(f"   - Watermark: ~{watermark_time:.2f} s")
    
    report_lines.append(f"\n4. Cache Statistics:")
    report_lines.append(f"   - Baseline Real Cache Hit Rate: {baseline_cache['real_rate']:.2f}% (Hits: {baseline_cache['real_hits']}/{baseline_cache['total']})")
    report_lines.append(f"   - Watermark Real Cache Hit Rate: {watermark_cache['real_rate']:.2f}% (Hits: {watermark_cache['real_hits']}/{watermark_cache['total']})")
    report_lines.append(f"   - Baseline Fake Cache Hit Rate: {baseline_cache['fake_hit_rate']:.2f}% (Hits: {baseline_cache['fake_hits']}/{baseline_cache['fake_hits'] + baseline_cache['fake_new']})")
    report_lines.append(f"   - Watermark Fake Cache Hit Rate: {watermark_cache['fake_hit_rate']:.2f}% (Hits: {watermark_cache['fake_hits']}/{watermark_cache['fake_hits'] + watermark_cache['fake_new']})")

    report_lines.append(f"\n5. Process Score (ToolEval Heuristic):")
    report_lines.append(f"   Formula: Score = (SuccessCalls * 10) + (UniqueTools * 5) - (5 * log(Steps))")
    if baseline_process_scores:
        report_lines.append(f"   - Baseline Mean Score: {np.mean(baseline_process_scores):.2f}")
    if watermark_process_scores:
        report_lines.append(f"   - Watermark Mean Score: {np.mean(watermark_process_scores):.2f}")

    # decoding summary loading moved up to affect stats
    decode_summary_path = os.path.join(base_dir, "decode", "summary.json")
    decode_summary = None
    if os.path.exists(decode_summary_path):
        try:
            decode_summary = load_json(decode_summary_path)
            # Update watermark stats from accurate decoding results
            if "results" in decode_summary:
                new_bits = []
                for res in decode_summary["results"]:
                    # Prefer extracted_packets (RLNC raw count) > total_bits (verified stream len)
                    if "extracted_packets" in res:
                        new_bits.append(len(res["extracted_packets"]))
                    else:
                        new_bits.append(res.get("total_bits", 0))
                
                # Only replace if meaningful
                if new_bits and sum(new_bits) > 0:
                    watermark_stats["bits"] = new_bits
                    print(f"[INFO] Updated embedded bits stats using decode summary (Count: {sum(new_bits)})")
        except Exception as e:
            print(f"Error reading decode summary for stats: {e}")

    report_lines.append(f"\n6. Embedded Bits Analysis (Watermark Only):")
    if watermark_stats['bits']:
        report_lines.append(f"   - Mean: {np.mean(watermark_stats['bits']):.2f} bits")
        report_lines.append(f"   - Max: {np.max(watermark_stats['bits'])} bits")
        report_lines.append(f"   - Min: {np.min(watermark_stats['bits'])} bits")
        report_lines.append(f"   - Zero Bits Count: {watermark_stats['bits'].count(0)} / {len(watermark_stats['bits'])}")
    else:
        report_lines.append("   - No watermark data found.")

    # Decoding Accuracy Analysis
    if decode_summary:
         global_acc = decode_summary.get("global_accuracy", 0.0)
         total_checked = decode_summary.get("total_checked_bits", 0)
         rlnc_status = decode_summary.get("rlnc_status", "N/A")
         rlnc_stats = decode_summary.get("rlnc_stats", {})
         
         report_lines.append(f"\n7. Decoding Accuracy:")
         report_lines.append(f"   - Global Accuracy: {global_acc:.2f}%")
         report_lines.append(f"   - Total Checked Bits: {total_checked}")
         if rlnc_status != "N/A":
             report_lines.append(f"   - RLNC Recovery Status: {rlnc_status}")
             if rlnc_stats.get("loss_simulated"):
                  loss_rate = rlnc_stats.get("loss_ratio", 0) * 100
                  orig = rlnc_stats.get("original_packets", 0)
                  rem = rlnc_stats.get("remaining_packets", 0)
                  report_lines.append(f"     [Loss Simulation] Rate: {loss_rate:.1f}% | Packets: {orig} -> {rem}")
    
    report_content = "\n".join(report_lines)
    print(report_content)
    
    output_path = os.path.join(base_dir, "analysis_summary.txt")
    with open(output_path, "w") as f:
        f.write(report_content)
    print(f"\n[INFO] Report saved to: {output_path}")
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        # Default to the latest run in output/toolbench_predictions
        root_output = "output/toolbench_predictions"
        if os.path.exists(root_output):
            dirs = [os.path.join(root_output, d) for d in os.listdir(root_output) if os.path.isdir(os.path.join(root_output, d))]
            dirs.sort(key=os.path.getmtime)
            base_dir = dirs[-1] if dirs else "."
        else:
            print("Error: No output directory found. Please specify a path.")
            sys.exit(1)
    
    print(f"Analyzing directory: {base_dir}")
    analyze_results(base_dir)
