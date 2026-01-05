"""ToolBench prediction output writer (ToolEval compatible)."""

import json
from pathlib import Path
from typing import Dict, List, Optional


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_answer_record(
    method: str,
    final_answer: str,
    trajectory: List[Dict],
    total_steps: int,
    query: str,
    available_tools: List[Dict],
    duration: float = 0.0,
    watermark_trace: Optional[List[Dict]] = None,
) -> Dict:
    record = {
        "method": method,
        "total_steps": total_steps,
        "final_answer": final_answer,
        "query": query,
        "available_tools": available_tools,
        "duration": duration,
        "answer_details": trajectory,
    }
    if watermark_trace:
        record["watermark_trace"] = watermark_trace
    return record


def save_prediction(
    run_dir: Path,
    test_set: str,
    query_id: str,
    record: Dict,
) -> Path:
    target_dir = run_dir / test_set
    _ensure_dir(target_dir)
    file_path = target_dir / f"{query_id}.json"
    with open(file_path, "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return file_path
