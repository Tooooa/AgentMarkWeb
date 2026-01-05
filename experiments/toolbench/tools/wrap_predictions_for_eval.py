"""
Wrap prediction JSONs to a ToolEval-friendly format by normalizing final_answer.

Usage:
  python wrap_predictions_for_eval.py --input_dir output/toolbench_predictions/smoke_baseline --output_dir output/toolbench_predictions/smoke_baseline_wrapped --return_type give_answer

Behavior:
  - Recursively scans input_dir for *.json
  - Rewrites final_answer -> {"return_type": <return_type>, "final_answer": <original_text>}
  - Copies other fields unchanged.
"""

import argparse
import json
from pathlib import Path


def wrap_file(src: Path, dst: Path, return_type: str) -> None:
    data = json.loads(src.read_text())
    original = data.get("final_answer", "")
    data["final_answer"] = {"return_type": return_type, "final_answer": original}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Prediction directory to wrap")
    parser.add_argument("--output_dir", required=True, help="Output directory to save wrapped files")
    parser.add_argument("--return_type", default="give_answer", help="Value for return_type field")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = list(input_dir.rglob("*.json"))
    if not files:
        raise SystemExit(f"No json files found under {input_dir}")

    for src in files:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        wrap_file(src, dst, args.return_type)

    print(f"[INFO] wrapped {len(files)} files to {output_dir}")


if __name__ == "__main__":
    main()
