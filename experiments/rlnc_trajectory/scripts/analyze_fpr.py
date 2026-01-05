
"""
RLNC False Positive Rate (FPR) and Forgery Analysis Script
- Loads watermarked traces and extracts received packets.
- Performs Monte-Carlo simulations to measure the probability of accidental decoding (FPR)
  under "Unwatermarked" (Random bits) and "Wrong Key" conditions.
- Metrics are reported as a function of 'Overhead' (k = m - n).
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentmark.core.rlnc_codec import DeterministicRLNC
try:
    from agentmark.core.watermark_sampler import differential_based_decoder
except ImportError:
    pass # might be needed if we re-decode, but we mostly need packet extraction logic from decode script

def load_json(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return json.load(f)

def extract_packets_from_dir(pred_dir: Path) -> List[Tuple[int, int]]:
    """
    Extracts all (index, bit_value) packets from valid watermark traces in the directory.
    Reuses logic from decode_toolbench_watermark.py essentially.
    """
    files = sorted(pred_dir.rglob("*.json"))
    all_packets = []
    
    print(f"[INFO] Scanning {len(files)} files in {pred_dir}...")
    
    for f in files:
        try:
            data = load_json(f)
        except:
            continue
            
        trace = data.get("watermark_trace", [])
        if not trace:
            continue
            
        for entry in trace:
            # We need effective_probs and chosen to decode bits locally 
            # OR we can trust the 'bit_index_before/after' if we trust the trace logic.
            # However, the trace doesn't explicitly store the *decoded bit value* usually, 
            # it stores the indices. We must re-decode the bit value from the chosen action 
            # to be sure what was "received".
            
            # Simplified: The run_experiment.py logic for Differential:
            # chosen, _, consumed_bits, _ = sample_behavior_differential(...)
            # The bit stream was embedded.
            
            # Note: The trace in run_experiment.py stores:
            # "bit_index_before": bit_before,
            # "bit_index_after": bit_index,
            # "chosen": chosen
            
            # We need to recover WHICH bits correspond to these indices.
            # In a real decoding scenario, we assume the bits are what the decoder sees.
            
            probs = entry.get("effective_probs")
            chosen = entry.get("chosen")
            context = entry.get("context_for_key") # Observation
            round_num = entry.get("task_idx", 0) + entry.get("round", 0)
            
            start = entry.get("bit_index_before")
            end = entry.get("bit_index_after")
            
            if start is None or end is None or not probs or not chosen:
                continue
                
            length = end - start
            if length <= 0:
                continue
                
            # Re-decode bits
            try:
                decoded_bits_str = differential_based_decoder(
                    probabilities=probs,
                    selected_behavior=chosen,
                    context_for_key=context,
                    round_num=round_num
                )
            except:
                continue
            
            # Align length
            if len(decoded_bits_str) != length:
                continue
                
            # Store packets
            for i, bit_char in enumerate(decoded_bits_str):
                # Packet: (Coefficient Index, Bit Value)
                idx = start + i
                val = int(bit_char)
                all_packets.append((idx, val))
                
    return all_packets

def solve_consistency(packets: List[Tuple[int, int]], encoder: DeterministicRLNC, payload_len: int) -> bool:
    """
    Checks if the system of linear equations defined by 'packets' is consistent.
    A * x = y
    
    For RLNC over GF(2), if rank(A|y) == rank(A), it is consistent.
    If the system is overdetermined (rows > cols) and consistent, it means likely the key is correct.
    If inconsistent, the key is wrong (or data is corrupted).
    """
    if not packets:
        return False
        
    # Unique packets only to avoid singular matrix issues due to duplicates
    # (Though duplicates don't hurt consistency check, just redundant)
    unique_packets = list(set(packets))
    
    # We need to build the matrix.
    # Rows = m packets
    # Cols = n payload bits
    
    m = len(unique_packets)
    n = payload_len
    
    # Optimization: If m < n, it's always "consistent" (under-determined) in the sense that solutions exist.
    # But for "Authentication/Verification", we strictly require m >= n to prove we recovered the payload.
    # Actually, for FPR, if m < n, we might find *many* solutions. 
    # The "Acceptance Rule" usually implies: "Unique Solution Exists AND is Consistent".
    # So we should strictly check: Rank(A) == n AND Consistent.
    
    matrix = []
    vector = []
    
    for idx, val in unique_packets:
        coeffs = encoder._generate_coeffs(idx) # This uses the encoder's internal stream_key
        matrix.append(coeffs)
        vector.append(val)
        
    A = np.array(matrix, dtype=int)
    y = np.array(vector, dtype=int)
    
    # Gaussian Elimination over GF(2) to check consistency
    # We can reuse the solver or write a quick rank check.
    # A fast way is to augment A with y and check rank.
    
    # rank(A)
    # rank(A|y)
    # Consistent iff rank(A) == rank(A|y)
    
    # Note: numpy.linalg.matrix_rank is for real/complex numbers. We need GF(2) rank.
    # We will use the Gaussian elimination method from rlnc_codec logic.
    
    rows, cols = A.shape
    augmented = np.hstack((A, y.reshape(-1, 1)))
    
    # Perform Gaussian Elimination on Augmented Matrix
    pivot_row = 0
    pivot_cols = []
    
    # Work on a copy to be safe (though we build fresh every time here)
    mat = augmented.copy()
    
    h_rows, h_cols = mat.shape
    
    for c in range(h_cols - 1): # Don't pivot on the last column (y vector) yet
        if pivot_row >= h_rows:
            break
            
        # Find pivot
        candidates = [r for r in range(pivot_row, h_rows) if mat[r, c] == 1]
        if not candidates:
            continue
            
        curr = candidates[0]
        # Swap
        mat[[pivot_row, curr]] = mat[[curr, pivot_row]]
        
        # Eliminate
        # Vectorized XOR on rows below
        # Find rows below pivot_row that have 1 in this col
        # (This is slightly slow in Python loops but okay for logic)
        for r in range(pivot_row + 1, h_rows):
            if mat[r, c] == 1:
                mat[r] ^= mat[pivot_row]
                
        pivot_row += 1
        
    # After REF (Row Echelon Form):
    # Check for consistency:
    # If there is a row where all A-part is 0 but y-part is 1, -> Inconsistent (0 != 1)
    
    # The A-part cols are 0 to n-1. The y-part is col n.
    for r in range(h_rows):
        # Check if row is all zeros in A part
        if not np.any(mat[r, :-1]):
            # If A part is all zero, y must be 0
            if mat[r, -1] == 1:
                return False # Inconsistent! 0x = 1

    return True # Consistent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--pred_dirs", nargs='+', help="List of prediction directories")
    parser.add_argument("--test_n", type=int, help="Payload length N (default: 32)")
    parser.add_argument("--trials", type=int, help="Trials per k (default: 100)")
    parser.add_argument("--max_k", type=int, help="Max k (default: 16)")
    parser.add_argument("--step_k", type=int, help="Step k (default: 2)")
    parser.add_argument("--output_dir", type=str, help="Directory to save results")
    args = parser.parse_args()
    
    # Defaults
    config = {
        "pred_dirs": [],
        "test_n": 32,
        "trials": 100,
        "max_k": 16,
        "step_k": 2,
        "output_dir": "."
    }
    
    # Load Config if provided
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            print(f"[INFO] Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                loaded_cfg = json.load(f)
                config.update(loaded_cfg)
        else:
            print(f"[WARN] Config file not found: {args.config}")
            
    # CLI Overrides
    if args.pred_dirs: config["pred_dirs"] = args.pred_dirs
    if args.test_n: config["test_n"] = args.test_n
    if args.trials: config["trials"] = args.trials
    if args.max_k: config["max_k"] = args.max_k
    if args.step_k: config["step_k"] = args.step_k
    if args.output_dir: config["output_dir"] = args.output_dir
    
    if not config["pred_dirs"]:
        parser.error("No prediction directories specified (via --config or --pred_dirs)")
        
    n = config["test_n"]
    
    # 2. Extract Indices Only
    real_packets = []
    print(f"[INFO] Extracting packets from {len(config['pred_dirs'])} directories...")
    for d_str in config["pred_dirs"]:
        d_path = Path(d_str)
        if not d_path.exists():
            print(f"[WARN] Directory not found: {d_path}")
            continue
        print(f"  -> Scanning {d_path}...")
        packets = extract_packets_from_dir(d_path)
        real_packets.extend(packets)
        print(f"     Found {len(packets)} packets.")
    
    # We only care about indices for simulation distribution
    available_indices = list(set([p[0] for p in real_packets]))
    total_available = len(available_indices)
    print(f"[INFO] Total unique indices available: {total_available}")
    
    if total_available < n:
        print(f"[ERROR] Not enough indices to simulate N={n}. Found {total_available}.")
        return

    # 3. Simulate Ground Truth
    # Generate a random payload x
    # Generate 'valid' values y for the available indices using Correct Key
    print(f"[INFO] Simulating ground truth for N={n}...")
    correct_key = 42
    encoder_correct = DeterministicRLNC("0" * n, stream_key=correct_key)
    
    # Random hidden payload (we don't need to know it, just need consistent y)
    # Actually we DO need it to generate y = A x
    # Let's generate random x
    x_vec = [random.randint(0, 1) for _ in range(n)]
    
    # Generate valid y for all available indices
    # valid_packets_pool = [(idx, val), ...]
    valid_packets_pool = []
    
    for idx in available_indices:
        # Generate row
        coeffs = encoder_correct._generate_coeffs(idx)
        # y = dot(coeffs, x)
        val = 0
        for c, x_bit in zip(coeffs, x_vec):
            val ^= (c & x_bit)
        valid_packets_pool.append((idx, val))
        
    
    # 4. Analysis Loop
    results = {
        "n": n,
        "trials": config["trials"],
        "data": []
    }
    
    print(f"{'Overhead (k)':<15} | {'m':<10} | {'FPR (Rand Data)':<20} | {'FPR (Wrong Key)':<20}")
    print("-" * 70)
    
    # Simulation Encoders
    wrong_key = correct_key + 999
    encoder_wrong = DeterministicRLNC("0"*n, stream_key=wrong_key)
    
    for k in range(0, config["max_k"] + 1, config["step_k"]):
        m = n + k
        if m > total_available:
            print(f"[WARN] Skipping k={k} (m={m}), not enough data.")
            break
            
        consistent_rand_count = 0
        consistent_wrong_key_count = 0
        
        for _ in range(config["trials"]):
            # Experiment A: Unwatermarked Data (Random y)
            fake_packets = generate_fake_packets(available_indices, m)
            if solve_consistency(fake_packets, encoder_correct, n):
                consistent_rand_count += 1
                
            # Experiment B: Wrong Key (Valid y, Wrong A)
            sample_gt = random.sample(valid_packets_pool, m)
            if solve_consistency(sample_gt, encoder_wrong, n):
                consistent_wrong_key_count += 1
                
        
        fpr_rand = consistent_rand_count / config["trials"]
        fpr_wrong = consistent_wrong_key_count / config["trials"]

        # Calculate Stats (Mean, Std)
        import math
        
        sem_rand = math.sqrt(fpr_rand * (1 - fpr_rand) / config["trials"])
        sem_wrong = math.sqrt(fpr_wrong * (1 - fpr_wrong) / config["trials"])

        print(f"{k:<15} | {m:<10} | {fpr_rand:<.4f} +/- {sem_rand:<.4f} | {fpr_wrong:<.4f} +/- {sem_wrong:<.4f}")
        
        results["data"].append({
            "k": k,
            "m": m,
            "fpr_unwatermarked": fpr_rand,
            "fpr_unwatermarked_sem": sem_rand,
            "fpr_wrong_key": fpr_wrong,
            "fpr_wrong_key_sem": sem_wrong
        })
        
    # Save Results
    target_dir = Path(config["output_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"fpr_analysis_data_N{n}.json"
    
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {out_file.resolve()}")

if __name__ == "__main__":
    main()
