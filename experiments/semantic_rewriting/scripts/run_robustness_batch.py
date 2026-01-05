
import os
import sys
import glob
import argparse
import subprocess
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def run_single_task(task_path, bits_path, output_dir):
    try:
        # Construct command
        # Re-use the existing single-task script
        cmd = [
            sys.executable,
            "scripts/robustness_test.py",
            "--task", task_path,
            "--bits", bits_path,
            "--output", output_dir,
            "--steps", "9999" # Run all steps
        ]
        
        # Run process
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return False, f"Error in {task_path}: {result.stderr}"
        
        return True, f"Success: {task_path}"
        
    except Exception as e:
        return False, f"Exception creating process for {task_path}: {e}"

def aggregate_results(output_dir):
    print(f"\nAggregating results from {output_dir}...")
    report_files = glob.glob(os.path.join(output_dir, "report_*.json"))
    
    total_steps = 0
    total_matches = 0
    kl_sum = 0
    kl_count = 0
    
    steps_with_bit_mismatch = 0
    total_watermarked_steps = 0 # Steps where bits_orig > 0
    
    task_stats = []

    for rf in report_files:
        try:
            with open(rf, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                # Calculate per task stats
                t_matches = sum(1 for s in data if s.get('action_match', False))
                t_steps = len(data)
                
                if t_steps > 0:
                    task_stats.append({
                        "file": os.path.basename(rf),
                        "match_rate": t_matches / t_steps,
                        "steps": t_steps
                    })
                
                total_steps += t_steps
                total_matches += t_matches
                
                for step in data:
                    kl = step.get('kl', 0)
                    # Filter out inf or nan if any
                    if kl != float('inf') and kl == kl: # simple nan check
                       kl_sum += kl
                       kl_count += 1
                    
                    # Bit consistency check
                    bits_orig = step.get('bits_orig', 0)
                    bits_pert = step.get('bits_pert', 0)
                    
                    if bits_orig > 0:
                        total_watermarked_steps += 1
                        if bits_orig != bits_pert:
                            steps_with_bit_mismatch += 1

        except Exception as e:
            print(f"Error reading {rf}: {e}")

    print("\n" + "="*40)
    print("BATCH EXPERIMENT RESULTS SUMMARY")
    print("="*40)
    print(f"Total Tasks Analyzed: {len(report_files)}")
    print(f"Total Steps Analyzed: {total_steps}")
    if total_steps > 0:
        print(f"Overall Action Match Rate: {total_matches/total_steps:.2%} ({total_matches}/{total_steps})")
    
    if kl_count > 0:
        print(f"Average KL Divergence: {kl_sum/kl_count:.4f}")
        
    if total_watermarked_steps > 0:
        print(f"Bit Count Consistency: {1 - (steps_with_bit_mismatch/total_watermarked_steps):.2%} (Steps with encoded bits changed: {steps_with_bit_mismatch}/{total_watermarked_steps})")
        
    print("\nTask Match Rate Distribution:")
    # Simple histogram
    buckets = [0, 0.25, 0.5, 0.75, 1.0]
    counts = [0, 0, 0, 0, 0] # 0-0.25, 0.25-0.5, ...
    
    for t in task_stats:
        rate = t['match_rate']
        if rate == 1.0:
            counts[4] += 1
        elif rate >= 0.75:
            counts[3] += 1
        elif rate >= 0.5:
            counts[2] += 1
        elif rate >= 0.25:
            counts[1] += 1
        else:
            counts[0] += 1
            
    print(f"  100% Match: {counts[4]}")
    print(f"  75-99% Match: {counts[3]}")
    print(f"  50-74% Match: {counts[2]}")
    print(f"  25-49% Match: {counts[1]}")
    print(f"  0-24% Match:  {counts[0]}")
    print("="*40)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", default="data/")
    parser.add_argument("--bits_file", default="data/decoded_bits.json")
    parser.add_argument("--output_dir", default="output/")
    parser.add_argument("--concurrency", type=int, default=100)
    
    args = parser.parse_args()
    
    task_files = glob.glob(os.path.join(args.logs_dir, "*.json"))
    print(f"Found {len(task_files)} task logs.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for task_file in task_files:
            futures.append(executor.submit(run_single_task, task_file, args.bits_file, args.output_dir))
            
        print(f"Launched {len(futures)} tasks with {args.concurrency} concurrent threads...")
        
        completed = 0
        for f in tqdm(as_completed(futures), total=len(futures)):
            success, msg = f.result()
            if not success:
                print(f"\n{msg}")
            completed += 1
            
    # Aggregate
    aggregate_results(args.output_dir)

if __name__ == "__main__":
    main()
