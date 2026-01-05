
import os
import json
import glob
import sys
import re
from pathlib import Path
import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, PROJECT_ROOT)

from agentmark.core.local_llm import LocalLLMClient

def extract_full_response(text):
    """
    Extract the full generation (Thought + JSON).
    Stripping the Prompt is key.
    """
    # Try to strip the Prompt (by searching for the last 'assistant\n\n')
    if "assistant\n\n" in text:
        parts = text.split("assistant\n\n")
        # Take the last part, which is what the model actually generated
        return parts[-1].strip()
    
    # Simple Heuristic: If it contains characteristic words from the System Prompt, stripping failed.
    if "Cutting Knowledge Date:" in text:
        # Try to strip via the Example terminator
        if "}\nEnsure every candidate tool" in text:
             parts = text.split("output the final_answer there.")
             if len(parts) > 1:
                 return parts[-1].strip()
    
    return text.strip()

def verify_full_response(output_root):
    text_wm_config = {
        "ngram_len": 5,
        "keys": [654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
                 590, 639, 13, 715, 468, 990, 966, 226, 324, 585,
                 118, 504, 421, 521, 129, 669, 732, 225, 90, 960],
        "sampling_table_size": 65536,
        "sampling_table_seed": 0,
        "context_history_size": 1024,
        "detector_type": "weighted_mean",
        "threshold": 0.50,
        "watermark_mode": "distortionary",
        "num_leaves": 4
    }
    
    model_path = os.environ.get("LOCAL_MODEL_PATH", "meta-llama/Llama-3.2-3B-Instruct")
    client = LocalLLMClient(model_path=model_path, torch_dtype=torch.float16, watermark_config=text_wm_config)
    
    json_files = glob.glob(os.path.join(output_root, "**/*.json"), recursive=True)
    json_files = [f for f in json_files if "summary" not in f and "rlnc_meta" not in f]
    
    all_scores = []
    all_lengths = []
    all_steps = []
    
    print(f"\nAnalyzing {len(json_files)} files in {output_root} (Full Response)...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            trajectory = data.get("answer_details", [])
            steps_count = 0
            for step in trajectory:
                if isinstance(step, dict) and step.get("role") == "assistant":
                    steps_count += 1
                    full_text = step.get("message", "")
                    cleaned_text = extract_full_response(full_text)
                    
                    if len(cleaned_text) > 20:
                        res = client.synthid.detect_watermark(cleaned_text)
                        score = res.get('score', 0) if isinstance(res, dict) else (res[1] if isinstance(res, (list, tuple)) else 0)
                        all_scores.append(score)
                        all_lengths.append(len(cleaned_text))
                        
                        if len(all_scores) <= 3:
                            print(f"Debug - Head: {cleaned_text[:50]}... | Len: {len(cleaned_text)} | Score: {score:.4f}")
            
            if steps_count > 0:
                all_steps.append(steps_count)

        except:
            continue
            
    if all_scores:
        mean_score = np.mean(all_scores)
        mean_len = np.mean(all_lengths)
        mean_steps = np.mean(all_steps)
        print(f"Mean Score: {mean_score:.4f} | Std: {np.std(all_scores):.4f} | Count: {len(all_scores)}")
        print(f"Mean Length: {mean_len:.2f} chars")
        print(f"Mean Steps: {mean_steps:.2f} turns/task")
        detected = sum(1 for s in all_scores if s > 0.50)
        print(f"Detection Rate (T=0.50): {detected/len(all_scores)*100:.2f}%")
        return all_scores
    else:
        print("No valid responses found.")
        return []

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_full_response.py <baseline_dir> <watermark_dir>")
    else:
        b_dir = sys.argv[1]
        w_dir = sys.argv[2]
        
        print("\n=== BASELINE ANALYSIS ===")
        b_scores = verify_full_response(b_dir)
        
        print("\n=== WATERMARK ANALYSIS ===")
        w_scores = verify_full_response(w_dir)
        
        if b_scores and w_scores:
             print(f"\nDelta Mean: {np.mean(w_scores) - np.mean(b_scores):.4f}")
