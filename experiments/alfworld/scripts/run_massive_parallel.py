import os
import sys
import json
import random
import subprocess
import time
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Settings
MAX_CONCURRENT_PROCESSES = 60  # Max concurrent processes (keep below 100 to avoid overload)
ID_CONFIG = "configs/config.json"
OOD_CONFIG = "configs/config.json"
TOTAL_TASKS_PER_SET = 100
NUM_ROUNDS = 3
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_total_tasks(env_type="valid_unseen"):
    """
    Get total number of available tasks.
    env_type: 'valid_seen' (ID) or 'valid_unseen' (OOD)
    """
    base_path = os.path.expanduser("~/.cache/alfworld/json_2.1.1")
    data_path = os.path.join(base_path, env_type)

    # ALFWorld data layout: data_path/<type_folder>/<game_folder>/traj_data.json
    pattern = os.path.join(data_path, "**", "traj_data.json")
    game_files = glob.glob(pattern, recursive=True)

    return len(game_files)


def run_experiment_task(task_id, config_path, rounds, output_dir, log_file):
    """
    Run a single task experiment process.
    """
    cmd = [
        "python", os.path.join(CURRENT_DIR, "run_experiment.py"),
        "--config", config_path,
        "--eval-split", "id" if "ID" in output_dir else "ood",
        "--task-ids-list", str(task_id),
        "--num-rounds", str(rounds),
        "--output-dir", output_dir,
        "--log-level", "INFO",
        "--random-seed", "42"  # Fixed seed for reproducibility
    ]

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    try:
        with open(log_file, "w") as f:
            process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            return process.returncode
    except Exception as e:
        print(f"Task {task_id} failed with error: {e}")
        return 1


def check_task_completion(output_dir, expected_rounds):
    """
    Check whether a task has completed.
    Criteria: reports/<timestamp>/round_{expected_rounds} exists
    or reports/<timestamp>/aggregate/multi_round_summary.txt exists.
    """
    reports_base = os.path.join(output_dir, "reports")
    if not os.path.exists(reports_base):
        return False

    timestamp_dirs = glob.glob(os.path.join(reports_base, "*"))
    for ts_dir in timestamp_dirs:
        if not os.path.isdir(ts_dir):
            continue

        # Check last round report
        last_round_dir = os.path.join(ts_dir, f"round_{expected_rounds:02d}")
        if os.path.exists(last_round_dir):
            json_files = glob.glob(os.path.join(last_round_dir, "*.json"))
            if json_files:
                return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Run massive parallel ALFWorld experiment (ID + OOD)")
    parser.add_argument("--max-workers", type=int, default=MAX_CONCURRENT_PROCESSES, help="Max concurrent processes")
    parser.add_argument("--resume-dir", type=str, default=None, help="Directory to resume from")
    args = parser.parse_args()

    if args.resume_dir:
        base_output_dir = args.resume_dir
        if not os.path.exists(base_output_dir):
            print(f"Error: Resume directory {base_output_dir} does not exist.")
            sys.exit(1)
        print(f"Resuming experiment from: {base_output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Unified output root (relative to project root)
        global_output_root = "output/alfworld"
        base_output_dir = os.path.join(global_output_root, timestamp)
        os.makedirs(base_output_dir, exist_ok=True)

    print("=== Massive Parallel Experiment Start ===")
    print(f"Output: {base_output_dir}")
    print(f"Max Workers: {args.max_workers}")
    print(f"Rounds: {NUM_ROUNDS}")

    # Determine task list
    if args.resume_dir:
        # Resume mode: read task IDs from directory
        print("Scanning existing directories for task list...")

        # ID Tasks
        id_dirs = glob.glob(os.path.join(base_output_dir, "ID", "task_*"))
        id_tasks = []
        for d in id_dirs:
            try:
                t_id = int(os.path.basename(d).split("_")[1])
                id_tasks.append(t_id)
            except Exception:
                pass
        id_tasks.sort()

        # OOD Tasks
        ood_dirs = glob.glob(os.path.join(base_output_dir, "OOD", "task_*"))
        ood_tasks = []
        for d in ood_dirs:
            try:
                t_id = int(os.path.basename(d).split("_")[1])
                ood_tasks.append(t_id)
            except Exception:
                pass
        ood_tasks.sort()

        print(f"Resuming {len(id_tasks)} ID tasks and {len(ood_tasks)} OOD tasks found in directory.")

    else:
        # New run: random sampling (fixed seed for consistency)
        random.seed(42)

        # 1. Sample ID tasks (hardcoded limit: 140)
        total_id = 140
        id_tasks = sorted(random.sample(range(total_id), min(total_id, TOTAL_TASKS_PER_SET)))
        print(f"Selected {len(id_tasks)} ID tasks (from Valid Seen: {total_id})")

        # 2. Sample OOD tasks (hardcoded limit: 134)
        total_ood = 134
        ood_tasks = sorted(random.sample(range(total_ood), min(total_ood, TOTAL_TASKS_PER_SET)))
        print(f"Selected {len(ood_tasks)} OOD tasks (from Valid Unseen: {total_ood})")

    # 3. Build task queue
    futures = []

    print("Launching tasks...")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit ID tasks
        for t_id in id_tasks:
            out_dir = os.path.join(base_output_dir, "ID", f"task_{t_id}")
            if check_task_completion(out_dir, NUM_ROUNDS):
                print(f"[Skip] ID Task {t_id} already completed.")
                continue

            log_file = os.path.join(out_dir, "run.log")
            futures.append(executor.submit(run_experiment_task, t_id, ID_CONFIG, NUM_ROUNDS, out_dir, log_file))

        # Submit OOD tasks
        for t_id in ood_tasks:
            out_dir = os.path.join(base_output_dir, "OOD", f"task_{t_id}")
            if check_task_completion(out_dir, NUM_ROUNDS):
                print(f"[Skip] OOD Task {t_id} already completed.")
                continue

            log_file = os.path.join(out_dir, "run.log")
            futures.append(executor.submit(run_experiment_task, t_id, OOD_CONFIG, NUM_ROUNDS, out_dir, log_file))

        print(f"All {len(futures)} tasks queued. Waiting for completion...")

        # Progress
        completed = 0
        total = len(futures)
        for f in as_completed(futures):
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"Progress: {completed}/{total} ({(completed/total)*100:.1f}%)")

    print("Experiment execution finished.")
    print(f"Results saved in: {base_output_dir}")
    print("Recommended: run analysis scripts separately for ID and OOD folders.")


if __name__ == "__main__":
    main()
