import argparse
import subprocess
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_task(cmd, task_name, log_file):
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            shell=True,
            executable='/bin/bash'
        )
        process.wait()
    return task_name, process.returncode

def main():
    parser = argparse.ArgumentParser(description="Run massive parallel ToolBench experiments.")
    parser.add_argument("--max_workers", type=int, default=100, help="Max concurrent processes.")
    args = parser.parse_args()

    # Configuration
    splits = [
        "G1_instruction", 
        "G1_category", 
        "G1_tool",
        "G2_category", 
        "G2_instruction", 
        "G3_instruction"
    ]
    
    rounds = [
        {"name": "v1", "seed": 42},
        {"name": "v2", "seed": 12345},
        {"name": "v3", "seed": 2024}
    ]
    
    # Define experiment types requested by user
    experiments = [
        {
            "name_prefix": "red_green",
            "args": "--sampling_strategy red_green" 
        },
        {
            "name_prefix": "repetition",
            "args": "--sampling_strategy differential --use_rlnc false"
        }
    ]
    
    tasks_per_split = 20
    base_log_dir = Path("output/logs/massive_run")
    # Clean implementation: archive old logs or just new dir? 
    # Just mkdir, overwrite happens naturally by filename
    base_log_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_script = "scripts/toolbench/run_experiment.py"
    
    futures = []
    
    print(f"[INFO] Starting Massive Parallel Execution with {args.max_workers} workers.")
    print(f"[INFO] Experiments: {[e['name_prefix'] for e in experiments]}")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for exp in experiments:
            exp_prefix = exp["name_prefix"]
            exp_args = exp["args"]
            
            for round_cfg in rounds:
                round_name = round_cfg["name"]
                seed = round_cfg["seed"]
                
                # output/toolbench_predictions/{prefix}_{round_name}
                final_run_name = f"{exp_prefix}_{round_name}"
                
                for split in splits:
                    for i in range(tasks_per_split):
                        # Unique identifier for log file
                        task_identifier = f"{exp_prefix}_{round_name}_{split}_{i}"
                        log_file = base_log_dir / f"{task_identifier}.log"
                        
                        cmd = (
                            f"python {experiment_script} "
                            f"--config configs/toolbench/pipeline_config.json "
                            f"--split {split} "
                            f"--task_index {i} "
                            f"--seed {seed} "
                            f"--run_name {final_run_name} "
                            f"{exp_args}"
                        )

                        futures.append(executor.submit(run_task, cmd, task_identifier, log_file))
                    
        print(f"[INFO] Submitted total {len(futures)} tasks. Waiting for completion...")
        
        completed = 0
        total = len(futures)
        failed_tasks = []
        
        for future in as_completed(futures):
            task_name, return_code = future.result()
            completed += 1
            if return_code != 0:
                print(f"[{completed}/{total}] [FAILED] {task_name} (Code: {return_code})")
                failed_tasks.append(task_name)
            else:
                if completed % 20 == 0:
                     print(f"[{completed}/{total}] [SUCCESS] Progress update...")

    print("[INFO] All tasks completed.")
    if failed_tasks:
        print(f"[WARN] {len(failed_tasks)} tasks failed. Check logs.")

if __name__ == "__main__":
    main()
