
import os
import json
import glob
import sys
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, PROJECT_ROOT)

from agentmark.core.local_llm import LocalLLMClient

def verify_toolbench_sythonid(output_root):
    # Configure Sythonid (must match experiment config)
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
    
    print("\nInitializing LocalLLMClient for detection...")
    client = LocalLLMClient(
        model_path=model_path,
        torch_dtype=torch.float16,
        watermark_config=text_wm_config
    )
    
    print(f"\nScanning for JSON reports in {output_root}...")
    json_files = glob.glob(os.path.join(output_root, "**/*.json"), recursive=True)
    json_files = [f for f in json_files if "rlnc_meta" not in f and "summary" not in f]
    
    print(f"Found {len(json_files)} prediction files.")
    
    total_actions = 0
    detected_count = 0
    all_scores = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            trajectory = data.get("answer_details", [])
            query_id = os.path.basename(json_file).replace(".json", "")
            
            if not trajectory or not isinstance(trajectory, list):
                print(f"Skipping {json_file}: no valid answer_details")
                continue
                
            print(f"\nAnalyzing Task {query_id} ({len(trajectory)} steps)...")
            
            task_scores = []
            
            for step in trajectory:
                if not isinstance(step, dict):
                    continue
                if step.get("role") == "assistant":
                    content = step.get("message", "")
                    if not content or len(content) < 10:
                        continue
                    
                    try:
                        res = client.synthid.detect_watermark(content)
                        
                        score = 0
                        is_wm = False
                        if isinstance(res, dict):
                            score = res.get('score', 0)
                            is_wm = res.get('is_watermarked', False)
                        elif isinstance(res, (tuple, list)):
                            is_wm = res[0]
                            score = res[1] if len(res) > 1 else 0
                        else:
                            is_wm = bool(res)
                            
                        print(f"  Length: {len(content)} | Score: {score:.4f} | Detected: {is_wm}")
                        
                        task_scores.append(score)
                        all_scores.append(score)
                        total_actions += 1
                        if is_wm:
                            detected_count += 1
                    except Exception as det_err:
                        print(f"  Detection error: {det_err}")
            
            if task_scores:
                avg = sum(task_scores) / len(task_scores)
                print(f"  -> Task Aggregated Score: {avg:.4f}")
                
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    if total_actions > 0:
        print(f"\nOverall Summary:")
        print(f"Total Assistant Messages: {total_actions}")
        print(f"Detected: {detected_count}")
        print(f"Rate: {detected_count/total_actions*100:.2f}%")
        print(f"Avg Score: {sum(all_scores)/len(all_scores):.4f}")
    else:
        print("No assistant messages found.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        base_dir = "output/toolbench_predictions"
        if os.path.exists(base_dir):
            all_dirs = sorted(glob.glob(os.path.join(base_dir, "*")), key=os.path.getmtime, reverse=True)
            if all_dirs:
                candidate = os.path.join(all_dirs[0], "exp_toolbench_local_sythonid")
                root = candidate if os.path.exists(candidate) else all_dirs[0]
            else:
                root = ""
        else:
            root = ""
            
    if root and os.path.exists(root):
        verify_toolbench_sythonid(root)
    else:
        print("Could not find meaningful output directory.")
