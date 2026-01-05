
import json
import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("MarkLLM"))

from agentmark.core.local_llm import LocalLLMClient

def collect_scores(dir_path, client):
    all_scores = []
    
    # Use glob to find all json files like verify script
    import glob
    json_files = glob.glob(os.path.join(dir_path, "**/*.json"), recursive=True)
    json_files = [f for f in json_files if "rlnc_meta" not in f and "summary" not in f]
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            trajectory = data.get("answer_details", [])
            if not trajectory or not isinstance(trajectory, list):
                continue
                
            for step in trajectory:
                if not isinstance(step, dict):
                    continue
                if step.get("role") == "assistant":
                    content = step.get("message", "")
                    if not content or len(content) < 50:
                        continue
                    
                    try:
                        res = client.synthid.detect_watermark(content)
                        score = res.get('score', 0) if isinstance(res, dict) else (res[1] if isinstance(res, (list, tuple)) else 0)
                        all_scores.append(score)
                    except:
                        continue
        except:
            continue
    return all_scores

def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_synthid_stats.py <baseline_dir> <watermark_dir>")
        return

    baseline_dir = sys.argv[1]
    watermark_dir = sys.argv[2]
    
    # SynthID Config
    text_wm_config = {
        "ngram_len": 5,
        "keys": [654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
                 590, 639, 13, 715, 468, 990, 966, 226, 324, 585,
                 118, 504, 421, 521, 129, 669, 732, 225, 90, 960],
        "sampling_table_size": 65536,
        "sampling_table_seed": 0,
        "context_history_size": 1024,
        "detector_type": "weighted_mean",
        "threshold": 0.52,
        "watermark_mode": "distortionary",
        "num_leaves": 4
    }
    
    model_path = os.environ.get("LOCAL_MODEL_PATH", "meta-llama/Llama-3.2-3B-Instruct")
    print("Initializing LocalLLMClient...")
    client = LocalLLMClient(model_path=model_path, watermark_config=text_wm_config, device="cuda")
    
    print(f"Analyzing {baseline_dir}...")
    baseline_scores = collect_scores(baseline_dir, client)
    
    print(f"Analyzing {watermark_dir}...")
    watermark_scores = collect_scores(watermark_dir, client)
    
    if not baseline_scores or not watermark_scores:
        print("No scores found.")
        return
    
    b_mean, b_std = np.mean(baseline_scores), np.std(baseline_scores)
    w_mean, w_std = np.mean(watermark_scores), np.std(watermark_scores)
    
    print("\n=== Statistical Summary ===")
    print(f"Baseline:  Mean={b_mean:.4f}, Std={b_std:.4f}, Count={len(baseline_scores)}")
    print(f"Watermark: Mean={w_mean:.4f}, Std={w_std:.4f}, Count={len(watermark_scores)}")
    print(f"Gap (Mean Diff): {w_mean - b_mean:.4f}")
    
    # Simple histogram simulation
    bins = [0.45, 0.47, 0.49, 0.50, 0.51, 0.52, 0.53, 0.55, 0.60, 1.0]
    b_counts, _ = np.histogram(baseline_scores, bins=bins)
    w_counts, _ = np.histogram(watermark_scores, bins=bins)
    
    print("\n=== Score Distribution (Histogram) ===")
    print("Range        | Baseline % | Watermark %")
    print("-------------|------------|------------")
    for i in range(len(bins)-1):
        b_pct = b_counts[i] / len(baseline_scores) * 100
        w_pct = w_counts[i] / len(watermark_scores) * 100
        print(f"{bins[i]:.2f}-{bins[i+1]:.2f}  | {b_pct:10.2f}% | {w_pct:11.2f}%")

if __name__ == "__main__":
    main()
