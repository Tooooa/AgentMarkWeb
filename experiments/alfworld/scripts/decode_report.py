"""
ALFWorld watermark decode utility.
Reads evaluation_report JSON, rebuilds step probabilities, and runs the
Differential decoder to recover embedded bit streams.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

from agentmark.core.watermark_sampler import differential_based_decoder


def decode_task_bits(task_result: Dict[str, Any]) -> Dict[str, Any]:
    """Decode bits for a single task using its detection_trace."""
    wm_stats = task_result.get('watermark_stats') or {}
    trace: List[Dict[str, Any]] = wm_stats.get('detection_trace') or []

    decoded_bits_segments = []
    errors = []

    for entry in trace:
        probs = entry.get('probabilities')
        if not probs:
            errors.append(
                f"Step {entry.get('step_num')} missing probability distribution; cannot decode"
            )
            continue

        action = entry.get('action')
        if not action:
            errors.append(f"Step {entry.get('step_num')} missing action field")
            continue

        context = entry.get('context_for_key')
        round_num = entry.get('round_num')
        if round_num is None:
            round_num = max(entry.get('step_num', 1) - 1, 0)

        try:
            bits = differential_based_decoder(
                probabilities=probs,
                selected_behavior=action,
                context_for_key=context,
                round_num=round_num
            )
        except Exception as exc:
            errors.append(f"Step {entry.get('step_num')} decode failed: {exc}")
            continue

        decoded_bits_segments.append(bits)

    bit_stream = "".join(decoded_bits_segments)
    return {
        'bit_stream': bit_stream,
        'segments': decoded_bits_segments,
        'errors': errors,
        'total_bits': len(bit_stream)
    }


def main():
    parser = argparse.ArgumentParser(description="Decode watermark bits from ALFWorld evaluation_report")
    parser.add_argument("--report", required=True, help="Path to evaluation_report_*.json")
    parser.add_argument("--task-index", type=int, default=None, help="Decode a specific task index (0-based)")
    parser.add_argument("--output", help="Optional output file for decoded results")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    with report_path.open('r', encoding='utf-8') as f:
        report = json.load(f)

    watermarked_results = report.get('watermarked_results') or []
    if not watermarked_results:
        raise ValueError("Report has no watermarked_results; cannot decode")

    target_indices = (
        [args.task_index] if args.task_index is not None else range(len(watermarked_results))
    )

    outputs = []
    for idx in target_indices:
        if idx < 0 or idx >= len(watermarked_results):
            raise IndexError(f"Task index {idx} out of range (total {len(watermarked_results)})")

        task_result = watermarked_results[idx]
        decode_info = decode_task_bits(task_result)
        outputs.append({
            'task_index': idx,
            'task_id': task_result.get('task_id'),
            'task_type': task_result.get('task_type'),
            **decode_info
        })

    # Print to console
    for item in outputs:
        print("=" * 60)
        print(f"Task index: {item['task_index']}  (task_id={item.get('task_id')}, type={item.get('task_type')})")
        print(f"Total bits: {item['total_bits']}")
        if item['bit_stream']:
            preview = item['bit_stream'][:64]
            if len(item['bit_stream']) > 64:
                preview += "..."
            print(f"Bit stream preview: {preview}")
        else:
            print("Bit stream is empty")

        if item['errors']:
            print("Warnings/Errors:")
            for err in item['errors']:
                print(f"  - {err}")

    if args.output:
        output_path = Path(args.output)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
        print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
