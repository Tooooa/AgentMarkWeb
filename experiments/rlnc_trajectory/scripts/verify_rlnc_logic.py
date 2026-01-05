
import json
import sys
from pathlib import Path
import random

# Add project root
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agentmark.core.rlnc_codec import DeterministicRLNC
from agentmark.core.watermark_sampler import sample_behavior_differential

def main():
    run_dir = Path("output/toolbench_predictions/verify_rlnc_run")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # RLNC Setup
    meta = {"payload": "10101010", "stream_key": 42, "is_rlnc": True}
    (run_dir / "rlnc_meta.json").write_text(json.dumps(meta))
    
    encoder = DeterministicRLNC(meta["payload"], stream_key=meta["stream_key"])
    
    # Generate infinite bit stream from encoder to embed
    # We need to simulate step by step.
    
    trace = []
    bit_index = 0
    full_stream_check = ""
    
    # Mock efficient probs
    probs = {f"tool_{i}": 0.1 for i in range(10)} # 10 tools, uniform
    
    for i in range(20): # 20 steps
        # Get stream segment (we assume infinite availability)
        # Sampler needs the FULL stream or access to it? 
        # differential sampler takes `bit_stream` string and `bit_index`.
        # We need to pre-generate a large enough chunk.
        
        # Determine bit stream chunk needed?
        # Sampler reads variable amount. 
        # Let's provide a huge chunk starting from bit_index
        huge_chunk = encoder.get_stream(0, 1000) # Get first 1000 bits
        
        context = f"dummy_context_step_{i}"
        
        chosen, chosen_idx, consumed, info = sample_behavior_differential(
            probs, 
            huge_chunk,
            bit_index,
            context,
            round_num=i
        )
        
        entry = {
            "task_idx": 0,
            "round": i,
            "mode": "watermark",
            "bit_index_before": bit_index,
            "bit_index_after": bit_index + consumed,
            "effective_probs": probs,
            "chosen": chosen,
            "context_for_key": context
        }
        trace.append(entry)
        
        bit_index += consumed
        
    print(f"[INFO] Generated trace with {bit_index} bits embedded.")
    
    pred_data = {
        "query": "Smart Synthetic Verification Task",
        "watermark_trace": trace,
        "total_steps": 20
    }
    
    (run_dir / "dummy_task_0.json").write_text(json.dumps(pred_data, indent=2))

if __name__ == "__main__":
    main()
