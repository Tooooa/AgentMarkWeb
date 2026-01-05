#!/usr/bin/env python3
"""
ToolBench Experiment Analysis Script
Analyzes experiment results from a single run directory.

Usage:
    python generate_detailed_report.py /path/to/run_dir
    python generate_detailed_report.py  # Auto-detect latest run
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import statistics
from typing import Dict, List, Optional, Any
from datetime import datetime

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PRED_ROOT = ROOT / "output" / "toolbench_predictions"


def find_latest_run_dir(pred_root: Path) -> Optional[Path]:
    """Find the most recent run directory by modification time."""
    if not pred_root.exists():
        return None
    
    candidates = []
    for name in os.listdir(pred_root):
        run_path = pred_root / name
        if not run_path.is_dir():
            continue
        # Check if it's a valid run directory (has exp_* subdirs or eval dir)
        if any((run_path / marker).exists() for marker in ["exp_baseline", "exp_watermark", "eval", "converted"]):
            candidates.append(run_path)
    
    if not candidates:
        return None
    
    # Sort by modification time, newest first
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_exp_dirs(run_dir: Path) -> Dict[str, Path]:
    """Find all exp_* directories in the run directory."""
    exp_dirs = {}
    for item in run_dir.iterdir():
        if item.is_dir() and item.name.startswith("exp_"):
            # Classify as baseline or watermark
            name_lower = item.name.lower()
            if "baseline" in name_lower:
                exp_dirs["baseline"] = item
            elif any(kw in name_lower for kw in ["watermark", "synthid", "differential", "rlnc"]):
                exp_dirs["watermark"] = item
            else:
                exp_dirs[item.name] = item
    return exp_dirs


def load_all_eval_results(run_dir: Path) -> Dict[str, bool]:
    """
    Load all results from eval/*.json into a mapping.
    Key: (split, mode_suffix, query_id) -> is_solved
    """
    eval_map = {}
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        return {}
    
    for eval_file in eval_dir.glob("*.json"):
        # File name format: {split}_{mode}.json
        name = eval_file.stem
        if "_" not in name:
            continue
        
        parts = name.rsplit("_", 1)
        split = parts[0]
        mode_suffix = parts[1] # baseline or watermark
        
        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)
            for qid, info in data.items():
                is_solved_dict = info.get("is_solved", {})
                if isinstance(is_solved_dict, dict):
                    solved = any("Solved" in str(v) for v in is_solved_dict.values())
                else:
                    solved = "Solved" in str(is_solved_dict)
                eval_map[(split, mode_suffix, qid)] = solved
        except:
            continue
    return eval_map


def extract_metrics_from_prediction(filepath: Path, mode_type: str, split: str, eval_map: Dict) -> Optional[Dict[str, Any]]:
    """Extract metrics from a single prediction JSON file."""
    try:
        qid = filepath.stem
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 1. Use Eval Map if available
        is_solved = None
        if (split, mode_type, qid) in eval_map:
            is_solved = eval_map[(split, mode_type, qid)]
        
        # 2. Heuristic fallback
        if is_solved is None:
            final_answer = data.get("final_answer", {})
            has_give_answer = final_answer.get("return_type") == "give_answer"
            
            # Check for Finish in trajectory
            answer_details = data.get("answer_details", [])
            has_finish = False
            for step in answer_details:
                if not isinstance(step, dict): continue
                msg = step.get("message", "")
                if not isinstance(msg, str): continue
                
                # Check for "Finish" as a chosen action in action_weights or ReAct format
                if '"Finish"' in msg or "'Finish'" in msg or "Action: Finish" in msg:
                    has_finish = True
                    break
            is_solved = has_give_answer and has_finish

        # Extract bits and steps
        steps = data.get("total_steps", 0)
        watermark_trace = data.get("watermark_trace", [])
        bits_embedded = sum(t.get("bit_index_after", 0) - t.get("bit_index_before", 0) 
                          for t in watermark_trace if isinstance(t, dict))
        
        return {
            "query_id": qid,
            "solved": is_solved,
            "steps": steps,
            "bits_embedded": bits_embedded
        }
    except:
        return None


def collect_metrics(exp_dirs: Dict[str, Path], eval_map: Dict) -> Dict[str, Any]:
    """Collect all metrics across all experiment directories."""
    all_data = defaultdict(lambda: defaultdict(list))
    
    for mode, exp_dir in exp_dirs.items():
        mode_type = "baseline" if "baseline" in mode else "watermark"
        for split_dir in exp_dir.iterdir():
            if not split_dir.is_dir():
                continue
            split = split_dir.name
            for json_file in split_dir.glob("*.json"):
                m = extract_metrics_from_prediction(json_file, mode_type, split, eval_map)
                if m:
                    all_data[mode][split].append(m)
    return all_data


def compute_stats(metrics_list):
    if not metrics_list:
        return {"solved": 0, "total": 0, "rate": 0, "avg_steps": 0, "bits": 0}
    
    solved = sum(1 for m in metrics_list if m["solved"])
    total = len(metrics_list)
    steps = [m["steps"] for m in metrics_list if m["steps"] > 0]
    bits = sum(m["bits_embedded"] for m in metrics_list)
    
    return {
        "solved": solved,
        "total": total,
        "rate": (solved / total * 100) if total > 0 else 0,
        "avg_steps": statistics.mean(steps) if steps else 0,
        "bits": bits
    }


def generate_report(run_dir: Path) -> str:
    eval_map = load_all_eval_results(run_dir)
    exp_dirs = find_exp_dirs(run_dir)
    all_metrics = collect_metrics(exp_dirs, eval_map)
    
    report = [f"# ToolBench Report: `{run_dir.name}`\n", f"Generated: {datetime.now()}\n"]
    
    # Summary Table
    report.append("## 1. Success Rate Summary\n")
    all_splits = sorted(set(s for m in all_metrics.values() for s in m.keys()))
    modes = sorted(all_metrics.keys())
    
    header = "| Split | " + " | ".join(modes) + " |"
    sep = "|:---|" + "|".join(":---:" for _ in modes) + "|"
    report.append(header); report.append(sep)
    
    for split in all_splits:
        row = f"| {split} |"
        for mode in modes:
            stats = compute_stats(all_metrics[mode].get(split, []))
            row += f" {stats['solved']}/{stats['total']} ({stats['rate']:.1f}%) |"
        report.append(row)
    
    # Detailed Section
    report.append("\n## 2. Detailed Performance\n")
    for mode in modes:
        report.append(f"### Mode: `{mode}`")
        report.append("| Split | Rate | Steps | Bits |")
        report.append("|:---|:---:|:---:|:---:|")
        for split in sorted(all_metrics[mode].keys()):
            s = compute_stats(all_metrics[mode][split])
            report.append(f"| {split} | {s['rate']:.1f}% | {s['avg_steps']:.1f} | {s['bits']} |")
        report.append("")

    # Decode Results
    decode_file = run_dir / "decode" / "summary.json"
    if decode_file.exists():
        report.append("## 3. Watermark Decoding\n")
        with open(decode_file, 'r') as f:
            d = json.load(f)
        report.append(f"- Accuracy: {d.get('global_accuracy', 0):.1f}%")
        report.append(f"- Bits Decoded: {d.get('total_bits_decoded', 0)}")
        report.append(f"- RLNC: {d.get('rlnc_status', 'N/A')}")
        
    return "\n".join(report)


if __name__ == "__main__":
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_run_dir(DEFAULT_PRED_ROOT)
    if not run_dir:
        print("No run dir found"); sys.exit(1)
        
    report = generate_report(run_dir)
    with open(run_dir / "analysis_report.md", "w") as f:
        f.write(report)
    print(f"Report saved to {run_dir / 'analysis_report.md'}")
    print(report)
