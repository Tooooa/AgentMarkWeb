"""
Convert AgentMark predictions to ToolEval-friendly format.

Input:
  --pred_dir: directory containing per-query prediction JSONs
  --query_file: test instruction JSON (must contain query_id, query, api_list)
  --method: name recorded in output (default: agentmark)
  --output: path to save converted JSON

Output format:
[
  {
    "query": "...",
    "available_tools": [...],
    "answer": {
      "method": "...",
      "total_steps": N,
      "final_answer": "...",
      "answer_details": [...]   # keep raw trajectory
    }
  },
  ...
]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_queries(path: Path) -> Dict[str, dict]:
    data = json.loads(path.read_text())
    mapping = {}
    for item in data:
        qid = str(item.get("query_id"))
        mapping[qid] = item
    return mapping


def unwrap_final_answer(val: Any) -> str:
    if isinstance(val, dict):
        return val.get("final_answer", "")
    return str(val) if val is not None else ""


def build_chain(answer_details_raw: list, final_answer: str) -> list:
    """
    Chain assistant/tool entries with next pointers, then append Finish.
    """
    nodes = []
    for step in answer_details_raw or []:
        nodes.append(
            {
                "role": step.get("role", ""),
                "message": step.get("message", ""),
                "next": [],
            }
        )
    # Append Finish
    finish_msg = str({"name": "Finish", "arguments": {}, "response": final_answer})
    nodes.append({"role": "tool", "message": finish_msg, "next": []})
    # Link nodes
    for i in range(len(nodes) - 1):
        nodes[i]["next"] = [nodes[i + 1]]
    return nodes


def convert(pred_dir: Path, query_file: Path, method: str, output: Path) -> None:
    qmap = load_queries(query_file)
    items: Dict[str, Any] = {}
    for file in sorted(pred_dir.glob("*.json")):
        try:
            pred = json.loads(file.read_text())
        except Exception:
            continue
        qid = file.stem
        qinfo = qmap.get(qid, {})
        tools = []
        for t in qinfo.get("api_list", []):
            t_copy = dict(t)
            # ensure name field exists
            t_copy["name"] = t.get("api_name") or t.get("tool_name") or t.get("name", "")
            tools.append(t_copy)
        final_answer = unwrap_final_answer(pred.get("final_answer", ""))
        raw_details = pred.get("answer_details", [])
        answer = {
            "method": method or pred.get("method", ""),
            "total_steps": pred.get("total_steps", 0),
            "final_answer": final_answer,
            "answer_details": build_chain(raw_details, final_answer),
        }
        items[qid] = {
            "query": qinfo.get("query", ""),
            "available_tools": tools,
            "answer": answer,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    print(f"[INFO] converted {len(items)} predictions -> {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--query_file", required=True)
    ap.add_argument("--method", default="agentmark")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    convert(Path(args.pred_dir), Path(args.query_file), args.method, Path(args.output))


if __name__ == "__main__":
    main()
