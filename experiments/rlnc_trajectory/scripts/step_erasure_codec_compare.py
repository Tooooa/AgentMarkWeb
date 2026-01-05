"""
Minimum Comparison: RLNC vs No-EC vs Repetition (global step-level erasure)

Purpose: Responding to reviewer questions like "Can you get 100% without RLNC?"
Approach: Does not rerun the agent or modify action sequences; on the same watermarked trajectory, 
only the "payload encoding/decoding rules" are changed, and the recovery success rate of 
the three schemes is compared under the same step loss rate p.

Comparison Schemes (k=len(payload)):
- RLNC: Consistent with the existing implementation, the bit for packet_index=i is DeterministicRLNC.get_bit(i).
- No-EC: No error correction; cyclic bits of the original payload are sent: bit = payload[i % k].
        Decoding: Success if each position is observed at least once and without conflict.
- Repetition: Same as No-EC (cyclic bits); decoding uses majority voting (tie -> failure).

Note: The "packet budget" and loss pattern here completely reuse the packet_index (expanded from 
bit_index_before/after) generated from real trajectories. Thus, the comparison is on the difference 
in "encoding/decoding capability" under the same budget (does not rely on reinjecting watermarks).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentmark.core.rlnc_codec import DeterministicRLNC


@dataclass(frozen=True)
class StepIndices:
    step_id: int
    indices: Tuple[int, ...]

    @property
    def packet_count(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class Episode:
    query_id: str
    steps: Tuple[StepIndices, ...]

    @property
    def total_steps(self) -> int:
        return len(self.steps)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_rlnc_meta(pred_root: Path) -> Path:
    for c in [pred_root / "rlnc_meta.json", pred_root.parent / "rlnc_meta.json"]:
        if c.exists():
            return c
    cur = pred_root
    for _ in range(3):
        c = cur / "rlnc_meta.json"
        if c.exists():
            return c
        cur = cur.parent
    raise FileNotFoundError(f"Could not find rlnc_meta.json (searching up from {pred_root})")


def iter_episode_files(pred_root: Path) -> List[Path]:
    files = []
    for p in sorted(pred_root.rglob("*.json")):
        if p.name == "rlnc_meta.json":
            continue
        files.append(p)
    return files


def extract_episode(file: Path) -> Episode:
    data = load_json(file)
    trace = data.get("watermark_trace") or []
    steps: List[StepIndices] = []
    for entry in trace:
        step_id = int(entry.get("round", 0))
        start = entry.get("bit_index_before")
        end = entry.get("bit_index_after")
        if start is None or end is None:
            steps.append(StepIndices(step_id=step_id, indices=tuple()))
            continue
        s = int(start)
        e = int(end)
        if e <= s:
            steps.append(StepIndices(step_id=step_id, indices=tuple()))
            continue
        steps.append(StepIndices(step_id=step_id, indices=tuple(range(s, e))))
    return Episode(query_id=file.stem, steps=tuple(steps))


def simulate_iid_step_erasure(steps: Sequence[StepIndices], p: float, rng: random.Random) -> List[StepIndices]:
    return [s for s in steps if rng.random() >= p]


def collect_indices(steps: Sequence[StepIndices]) -> List[int]:
    out: List[int] = []
    for s in steps:
        out.extend(list(s.indices))
    return out


def wilson_95ci(success: int, total: int) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.96
    phat = success / total
    denom = 1 + (z**2) / total
    center = (phat + (z**2) / (2 * total)) / denom
    half = z * math.sqrt((phat * (1 - phat) + (z**2) / (4 * total)) / total) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def decode_noec(payload: str, indices: Sequence[int], bits: Sequence[int]) -> Optional[str]:
    k = len(payload)
    if k == 0:
        return None
    seen: Dict[int, int] = {}
    for idx, b in zip(indices, bits):
        j = idx % k
        if j in seen and seen[j] != b:
            return None
        seen[j] = b
    if len(seen) < k:
        return None
    return "".join(str(seen[j]) for j in range(k))


def decode_repetition(payload: str, indices: Sequence[int], bits: Sequence[int]) -> Optional[str]:
    k = len(payload)
    if k == 0:
        return None
    votes: Dict[int, List[int]] = {j: [] for j in range(k)}
    for idx, b in zip(indices, bits):
        votes[idx % k].append(int(b))
    out = []
    for j in range(k):
        if not votes[j]:
            return None
        c1 = sum(votes[j])
        c0 = len(votes[j]) - c1
        if c1 == c0:
            return None
        out.append("1" if c1 > c0 else "0")
    return "".join(out)


def build_bits(codec: str, payload: str, key: int, indices: Sequence[int]) -> Tuple[List[int], Optional[DeterministicRLNC]]:
    if codec == "rlnc":
        enc = DeterministicRLNC(payload, stream_key=key)
        return [int(enc.get_bit(i)) for i in indices], enc
    if codec in ("noec", "repetition"):
        k = len(payload)
        return [int(payload[i % k]) for i in indices], None
    raise ValueError(f"unknown codec: {codec}")


def rlnc_rank(indices: Sequence[int], enc: DeterministicRLNC) -> int:
    if not indices:
        return 0
    # Construct matrix directly using coefficient generator of RLNC and calculate GF(2) rank
    mat = np.array([enc._generate_coeffs(i) for i in indices], dtype=np.uint8)  # noqa: SLF001
    rows, cols = mat.shape
    rank = 0
    col = 0
    for r in range(rows):
        while col < cols and not mat[r:, col].any():
            col += 1
        if col >= cols:
            break
        pivot = r + int(np.argmax(mat[r:, col]))
        if mat[pivot, col] == 0:
            continue
        if pivot != r:
            mat[[r, pivot]] = mat[[pivot, r]]
        for rr in range(rows):
            if rr != r and mat[rr, col]:
                mat[rr] ^= mat[r]
        rank += 1
        col += 1
    return rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_roots", nargs="+", required=True, help="One or more watermark prediction root directories (to merge trajectories)")
    ap.add_argument("--output_dir", required=True, help="Output directory")
    ap.add_argument("--p_list", default="0,0.1,0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--codecs", default="rlnc,noec,repetition", help="Comma-separated: rlnc,noec,repetition")
    args = ap.parse_args()

    pred_roots = [Path(p) for p in args.pred_roots]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # meta consistency check
    payload = ""
    key = 0
    for pr in pred_roots:
        meta = load_json(find_rlnc_meta(pr))
        if not payload:
            payload = meta.get("payload", "")
            key = int(meta.get("stream_key", 0))
        else:
            if meta.get("payload", "") != payload or int(meta.get("stream_key", 0)) != key:
                raise ValueError(f"Inconsistent rlnc_meta at {pr}")
    if not payload:
        raise ValueError("Payload is empty")

    # episodes
    files: List[Path] = []
    for pr in pred_roots:
        files.extend(iter_episode_files(pr))
    episodes = [extract_episode(f) for f in sorted(files)]

    p_list = [float(x.strip()) for x in args.p_list.split(",") if x.strip()]
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    k = len(payload)

    rows: List[Dict[str, object]] = []
    for codec in codecs:
        for p in p_list:
            successes = 0
            total = 0
            recv_steps_sum = 0
            recv_packets_sum = 0
            # record rank (RLNC only) for explanation
            rank_sum = 0.0

            for r in range(args.trials):
                rng_master = random.Random(args.seed + r * 1000 + int(p * 1000))
                kept_all: List[StepIndices] = []
                for ep_idx, ep in enumerate(episodes):
                    rng_ep = random.Random(rng_master.randint(0, 2**31 - 1) + ep_idx * 17)
                    kept_all.extend(simulate_iid_step_erasure(ep.steps, p, rng_ep))
                indices = collect_indices(kept_all)
                bits, enc = build_bits(codec, payload, key, indices)

                if codec == "rlnc":
                    assert enc is not None
                    rank = rlnc_rank(indices, enc)
                    decoded = enc.decode(indices, bits) if rank >= k else None
                    ok = decoded == payload
                    rank_sum += rank
                elif codec == "noec":
                    decoded = decode_noec(payload, indices, bits)
                    ok = decoded == payload
                elif codec == "repetition":
                    decoded = decode_repetition(payload, indices, bits)
                    ok = decoded == payload
                else:
                    raise ValueError(codec)

                successes += int(bool(ok))
                total += 1
                recv_steps_sum += len(kept_all)
                recv_packets_sum += len(indices)

            lo, hi = wilson_95ci(successes, total)
            rows.append(
                {
                    "codec": codec,
                    "p": p,
                    "decode_success_rate": successes / total if total else 0.0,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "avg_received_steps": recv_steps_sum / total if total else 0.0,
                    "avg_received_packets": recv_packets_sum / total if total else 0.0,
                    "avg_rank": (rank_sum / total) if (total and codec == "rlnc") else "",
                    "k": k,
                    "trials": total,
                }
            )

    # Write CSV
    csv_path = out_dir / "compare_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["codec", "p", "decode_success_rate", "ci95_low", "ci95_high", "avg_received_steps", "avg_received_packets", "avg_rank", "k", "trials"],
        )
        w.writeheader()
        w.writerows(rows)

    # Write MD
    md = []
    md.append("# Baseline Comparison: RLNC vs No-EC vs Repetition (global step erasure)")
    md.append("")
    md.append(f"- k={k} (payload bits)")
    md.append(f"- trials R={args.trials}")
    md.append(f"- codecs={codecs}")
    md.append("")
    md.append("| codec | p | success_rate | 95% CI | avg_steps | avg_packets | avg_rank(RLNC) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['codec']} | {float(r['p']):.1f} | {r['decode_success_rate']*100:.2f}% | "
            f"[{r['ci95_low']*100:.2f}%, {r['ci95_high']*100:.2f}%] | "
            f"{float(r['avg_received_steps']):.2f} | {float(r['avg_received_packets']):.2f} | {r['avg_rank']} |"
        )
    md_path = out_dir / "compare_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    # Plot success vs p
    try:
        import matplotlib.pyplot as plt

        by_codec: Dict[str, List[Dict[str, object]]] = {}
        for r in rows:
            by_codec.setdefault(r["codec"], []).append(r)
        plt.figure(figsize=(7.4, 4.4))
        for codec, items in by_codec.items():
            items = sorted(items, key=lambda x: float(x["p"]))
            xs = [float(x["p"]) for x in items]
            ys = [float(x["decode_success_rate"]) for x in items]
            yerr_lo = [max(0.0, y - float(x["ci95_low"])) for x, y in zip(items, ys)]
            yerr_hi = [max(0.0, float(x["ci95_high"]) - y) for x, y in zip(items, ys)]
            plt.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="-o", capsize=3, label=codec)
        plt.ylim(-0.02, 1.02)
        plt.xlabel("step erasure p")
        plt.ylabel("decode_success_rate")
        plt.title("Codec Comparison under Step Erasures (global)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "compare_success_vs_p.png", dpi=200)
        plt.close()
    except Exception as e:  # noqa: BLE001
        (out_dir / "plot_error.txt").write_text(str(e), encoding="utf-8")

    print(f"[OK] saved -> {out_dir}")
    print(f"[OK] csv -> {csv_path}")
    print(f"[OK] md -> {md_path}")


if __name__ == "__main__":
    main()
