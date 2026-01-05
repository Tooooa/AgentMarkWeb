"""
ToolBench Watermark Decoding Script
- Reads watermark prediction files (containing watermark_trace) from the prediction directory
- Uses a differential watermark decoder to restore embedded bits for each task
- Outputs a decoding summary for subsequent verification
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentmark.core.watermark_sampler import differential_based_decoder


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def load_bit_stream(path: Optional[Path]) -> str:
    if path and path.exists():
        return path.read_text().strip()
    return ""


def decode_trace(trace: List[Dict[str, Any]], full_bit_stream: str = "") -> Dict[str, Any]:
    """Decode bits step-by-step for a single file's watermark_trace and verify accuracy."""
    segments = []
    errors = []
    
    total_checked_bits = 0
    matched_bits = 0
    
    for entry in trace:
        # The encoder only embeds when it is NOT Finish and the differential encoding actually consumes bits.
        # If bit_index_before == bit_index_after (or missing), no watermark bits were sent this round:
        # - Cannot perform verification
        # - Cannot treat bits inferred by the decoder as valid packets to avoid polluting RLNC equations
        start = entry.get("bit_index_before")
        end = entry.get("bit_index_after")
        embed_len = None
        if start is not None and end is not None:
            try:
                embed_len = int(end) - int(start)
            except Exception:  # noqa: BLE001
                embed_len = None

        if embed_len is not None and embed_len <= 0:
            segments.append("")
            continue

        probs = entry.get("effective_probs") or {}
        if not probs:
            errors.append(f"round={entry.get('round')} Missing probabilities")
            segments.append("")
            continue
        chosen = entry.get("chosen")
        if not chosen:
            errors.append(f"round={entry.get('round')} Missing chosen behavior")
            segments.append("")
            continue
        context = entry.get("context_for_key")
        # Encoder uses (task_idx + step_count) as round_num
        round_num = entry.get("task_idx", 0) + entry.get("round", 0)
        try:
            bits = differential_based_decoder(
                probabilities=probs,
                selected_behavior=chosen,
                context_for_key=context,
                round_num=round_num,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"round={round_num} Decoding failed: {exc}")
            segments.append("")
            continue

        # If embed_len is recorded, force alignment of decoded length; otherwise keep it as is
        if embed_len is not None and len(bits) != embed_len:
            errors.append(
                f"round={round_num} Decoded length inconsistent: decoded={len(bits)} expected={embed_len} (start={start}, end={end})"
            )
            # Conservative strategy: do not consider this round as valid bits to avoid polluting RLNC
            bits = ""

        segments.append(bits)
        
        # Verify bit accuracy
        if full_bit_stream and bits and embed_len:
            if start is not None and end is not None:
                # If bits were embedded, they must be within range
                if start < len(full_bit_stream) and end <= len(full_bit_stream):
                    expected = full_bit_stream[start:end]
                    # Bit-by-bit comparison
                    for b_dec, b_exp in zip(bits, expected):
                        total_checked_bits += 1
                        if b_dec == b_exp:
                            matched_bits += 1
                else:
                    errors.append(f"round={round_num} index out of stream range: {start}-{end} (len={len(full_bit_stream)})")

    bit_stream = "".join(segments)
    
    accuracy = 0.0
    if total_checked_bits > 0:
        accuracy = (matched_bits / total_checked_bits) * 100

    # Collect packets for RLNC
    extracted_packets = []
    for entry, seg_bits in zip(trace, segments):
        start = entry.get("bit_index_before")
        end = entry.get("bit_index_after")
        if start is None or end is None:
            continue
        try:
            embed_len = int(end) - int(start)
        except Exception:  # noqa: BLE001
            continue
        if embed_len <= 0:
            continue
        # Only accept segments with consistent length
        if not seg_bits or len(seg_bits) != embed_len:
            continue
        for i, bit in enumerate(seg_bits):
            extracted_packets.append((int(start) + i, int(bit)))
        
    return {
        "segments": segments,
        "bit_stream": bit_stream,
        "total_bits": len(bit_stream),
        "errors": errors,
        "verification": {
            "checked_bits": total_checked_bits,
            "matched_bits": matched_bits,
            "accuracy": accuracy
        },
        "extracted_packets": extracted_packets
    }


def decode_predictions(pred_dir: Path, bit_stream: str) -> List[Dict[str, Any]]:
    results = []
    for file in sorted(pred_dir.rglob("*.json")):
        try:
            data = load_json(file)
        except Exception as exc:  # noqa: BLE001
            results.append({"file": str(file), "error": f"Load failed: {exc}"})
            continue

        trace = data.get("watermark_trace")
        if not trace:
            results.append({"file": str(file), "warning": "Missing watermark_trace"})
            continue

        decode_res = decode_trace(trace, bit_stream)
        results.append(
            {
                "file": str(file),
                "query_id": file.stem,
                **decode_res,
            }
        )
    return results


def build_effective_bit_stream(
    base_message: str,
    config: dict,
    num_tasks: int,
) -> str:
    """Construct the final bitstream for the current run based on the embedding strategy (consistent with experiment)."""
    if not base_message:
        return base_message

    watermark_config = config.get("watermark_config", {})
    strategy = watermark_config.get("embedding_strategy", "one_time")
    if strategy != "cyclic":
        return base_message

    # Default max_steps for ToolBench
    max_steps = config.get("max_steps", 6)
    estimated_steps = max(1, num_tasks) * max_steps
    # Conservatively estimate 3 bits embedded per step in differential scheme
    estimated_bits_needed = max(estimated_steps * 3, len(base_message))
    repeats = max(1, (estimated_bits_needed // len(base_message)) + 2)
    effective_stream = base_message * repeats
    print(f"[INFO] Cyclic verification: expanded stream to {len(effective_stream)} bits (repeats {repeats})")
    return effective_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Simplified RLNC evaluation config JSON")
    ap.add_argument("--pipeline_config", default=None, help="Legacy pipeline config JSON (optional)")
    ap.add_argument("--pred_dir", default=None, help="Watermark prediction directory")
    ap.add_argument("--rlnc_meta", default=None, help="Explicit path to rlnc_meta.json")
    ap.add_argument("--output", default=None, help="Path to output summary")
    ap.add_argument("--bit_stream_path", default=None, help="Path to reference bit stream (if not RLNC)")
    ap.add_argument("--simulate_loss", action="store_true", help="Simulate packet loss")
    ap.add_argument("--loss_ratio", type=float, default=0.0, help="Packet loss ratio (0.0-1.0)")
    ap.add_argument("--scope", choices=["episode", "global"], default=None, help="Decoding scope")
    args = ap.parse_args()

    # Load Configs
    eval_cfg = load_json(Path(args.config)) if args.config else {}
    pipe_cfg = load_json(Path(args.pipeline_config)) if args.pipeline_config else {}

    # 1. Determine Prediction Directory
    pred_dir_str = args.pred_dir or eval_cfg.get("pred_dir") or pipe_cfg.get("watermark_specific", {}).get("run_name")
    if pred_dir_str:
        pred_dir = Path(pred_dir_str)
        # If it doesn't exist and is a simple name, try prepending the ToolBench output path
        if not pred_dir.exists() and not pred_dir.is_absolute() and "output/toolbench_predictions" not in pred_dir_str:
            pred_dir = Path("output") / "toolbench_predictions" / pred_dir_str
    else:
        pred_dir = Path("output") / "toolbench_predictions" / "exp_watermark"

    if not pred_dir.exists():
        print(f"[ERROR] Prediction directory not found: {pred_dir}")
        sys.exit(1)

    # 2. Locate rlnc_meta.json
    rlnc_meta_path = None
    if args.rlnc_meta:
        rlnc_meta_path = Path(args.rlnc_meta)
    elif eval_cfg.get("rlnc_meta_path"):
        rlnc_meta_path = Path(eval_cfg["rlnc_meta_path"])
    else:
        # Auto-detect logic
        candidates = [
            pred_dir / "rlnc_meta.json",
            pred_dir.parent / "rlnc_meta.json"
        ]
        for c in candidates:
            if c.exists():
                rlnc_meta_path = c
                break
    
    if rlnc_meta_path and not rlnc_meta_path.exists():
        print(f"[WARN] Specified rlnc_meta_path not found: {rlnc_meta_path}")
        rlnc_meta_path = None

    # 3. Determine Output Path
    if args.output:
        out_path = Path(args.output)
    elif eval_cfg.get("output_dir"):
        out_path = Path(eval_cfg["output_dir"]) / "decode_summary.json"
    else:
        out_path = pred_dir / "decode_summary.json"

    # 4. Decoding & Loss Configuration
    decode_params = eval_cfg.get("decode_params", {})
    decode_cfg = pipe_cfg.get("decode_config", {})
    
    simulate_loss = args.simulate_loss or decode_params.get("simulate_loss", False) or decode_cfg.get("simulate_packet_loss", False)
    loss_ratio = args.loss_ratio if args.loss_ratio > 0 else decode_params.get("loss_ratio", 0.0) or decode_cfg.get("packet_loss_ratio", 0.0)
    scope = args.scope or decode_params.get("scope") or "global"
    
    bit_path = args.bit_stream_path or decode_params.get("bit_stream_path") or decode_cfg.get("bit_stream_path") or pipe_cfg.get("watermark_specific", {}).get("bit_stream_path")
    bit_path = Path(bit_path) if bit_path else None
    
    # Load RLNC metadata
    rlnc_mode = False
    rlnc_payload = ""
    rlnc_key = 0
    if rlnc_meta_path and rlnc_meta_path.exists():
        try:
             meta = json.loads(rlnc_meta_path.read_text())
             if meta.get("is_rlnc"):
                 rlnc_mode = True
                 rlnc_payload = meta.get("payload", "")
                 rlnc_key = meta.get("stream_key", 0)
                 print(f"[INFO] Using metadata from: {rlnc_meta_path}")
                 print(f"[INFO] Detected RLNC mode. Payload len={len(rlnc_payload)}")
        except Exception as e:
             print(f"[WARN] Failed to load RLNC meta: {e}")

    # 5. Reconstruct/Load Bit Stream
    encoder = None
    if rlnc_mode:
        from agentmark.core.rlnc_codec import DeterministicRLNC
        encoder = DeterministicRLNC(rlnc_payload, stream_key=rlnc_key)
        bit_stream = encoder.get_stream(0, 50000) 
        print(f"[INFO] Reconstructed RLNC stream of length {len(bit_stream)} for checking.")
    else:
        raw_bit_stream = load_bit_stream(bit_path)
        json_files = list(pred_dir.rglob("*.json"))
        num_tasks = len(json_files)
        # Fallback values for effective stream construction
        wm_config = pipe_cfg.get("watermark_specific", {})
        full_wm_config = {**pipe_cfg.get("common_config", {}), **wm_config}
        bit_stream = build_effective_bit_stream(raw_bit_stream, full_wm_config, num_tasks)

    # 6. Decode and Recover
    decoded = decode_predictions(pred_dir, bit_stream)
    total_bits = sum(item.get("total_bits", 0) for item in decoded if isinstance(item, dict))
    
    # Calculate global accuracy (Verification against stream)
    total_checked = sum(item.get("verification", {}).get("checked_bits", 0) for item in decoded if isinstance(item, dict))
    total_matched = sum(item.get("verification", {}).get("matched_bits", 0) for item in decoded if isinstance(item, dict))
    global_accuracy = (total_matched / total_checked * 100) if total_checked > 0 else 0.0

    # RLNC Recovery Attempt
    # RLNC Recovery Attempt
    rlnc_status = "N/A"
    recovered_payload = ""
    rlnc_results = []
    
    if rlnc_mode:
        if scope == "global":
            full_packets = []
            for item in decoded:
                for idx, val in item.get("extracted_packets", []):
                    full_packets.append((idx, val))
            
            # De-dedup / Conflict resolution
            by_index = {}
            conflicts = set()
            for idx, val in full_packets:
                if idx in by_index:
                    if by_index[idx] != val: conflicts.add(idx)
                else: by_index[idx] = val
            for idx in conflicts: by_index.pop(idx, None)
            
            dedup_packets = list(by_index.items())
            print(f"[INFO] Global recovery: Collected {len(full_packets)} raw packets; dedup={len(dedup_packets)}.")

            # Loss Simulation (Global)
            if simulate_loss and loss_ratio > 0:
                random.seed(42)
                random.shuffle(dedup_packets)
                keep_count = int(len(dedup_packets) * (1 - loss_ratio))
                effective_packets = dedup_packets[:keep_count]
                print(f"[INFO] Global loss sim ({loss_ratio*100:.1f}%): remaining {len(effective_packets)} / {len(dedup_packets)}")
            else:
                effective_packets = dedup_packets

            all_indices = [p[0] for p in effective_packets]
            all_bits = [p[1] for p in effective_packets]
            
            try:
                recovered = encoder.decode(all_indices, all_bits)
                if recovered == rlnc_payload:
                    rlnc_status = "SUCCESS"
                    recovered_payload = recovered
                else:
                    rlnc_status = "FAILED"
            except Exception as e:
                rlnc_status = f"ERROR: {e}"
            
            print(f"[INFO] Global RLNC Recovery: {rlnc_status}")
            
        elif scope == "episode":
            success_count = 0
            for item in decoded:
                ep_packets = item.get("extracted_packets", [])
                
                # De-dedup per episode
                by_index = {}
                conflicts = set()
                for idx, val in ep_packets:
                    if idx in by_index:
                        if by_index[idx] != val: conflicts.add(idx)
                    else: by_index[idx] = val
                for idx in conflicts: by_index.pop(idx, None)
                dedup_ep = list(by_index.items())

                # Loss Simulation (Per episode)
                if simulate_loss and loss_ratio > 0:
                    random.seed(42 + hash(item.get("query_id", "")))
                    random.shuffle(dedup_ep)
                    keep_ep = int(len(dedup_ep) * (1 - loss_ratio))
                    effective_ep = dedup_ep[:keep_ep]
                else:
                    effective_ep = dedup_ep
                
                indices_ep = [p[0] for p in effective_ep]
                bits_ep = [p[1] for p in effective_ep]
                
                status_ep = "FAILED"
                try:
                    recovered = encoder.decode(indices_ep, bits_ep)
                    if recovered == rlnc_payload:
                        status_ep = "SUCCESS"
                        success_count += 1
                except: pass
                
                item["rlnc_recovery"] = {
                    "status": status_ep,
                    "packets_total": len(ep_packets),
                    "packets_dedup": len(dedup_ep),
                    "packets_effective": len(indices_ep)
                }
            
            rlnc_status = f"EPISODE_ACCURACY: {success_count}/{len(decoded)}"
            print(f"[INFO] Episode-level RLNC Recovery: {rlnc_status}")

    # (Previous variables needed for summary)
    rlnc_stats_final = {
        "loss_simulated": simulate_loss,
        "loss_ratio": loss_ratio,
        "scope": scope,
    }

    summary = {
        "pred_dir": str(pred_dir),
        "pred_files": len(decoded),
        "bit_stream_path": str(bit_path) if bit_path else "",
        "bit_stream_length": len(bit_stream),
        "total_bits_decoded": total_bits,
        "global_accuracy": global_accuracy,
        "total_checked_bits": total_checked,
        "rlnc_status": rlnc_status,
        "rlnc_stats": rlnc_stats_final,
        "results": decoded,
        "status": "ok",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[INFO] Saved decode summary -> {out_path}")


if __name__ == "__main__":
    main()
