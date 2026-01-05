"""
ToolBench Behavioral Watermark: RLNC Robustness Evaluation (step-level erasure)

Threat Model (Black-box): The adversary cannot tamper with actions, only cause some steps to be lost (erasure).
This script does not rerun the agent; it repeatedly simulates losses on the same original watermarked trajectory.

Input (automatically extracted from existing ToolBench watermark trajectories):
- Each episode (a query_id.json) contains watermark_trace (bit_index_before/after per step)
- RLNC metadata from rlnc_meta.json: payload=m, stream_key=K

RLNC Settings:
- Deterministic RLNC over GF(2): Coefficient vectors are deterministically generated from (K, packet_index)
- packet_index uses the "original encoding stream index" (corresponding to the indices expanded from bit_index_before/end in the trace)
  This is consistent with the definition of DeterministicRLNC on the encoding side, so the verifier can reconstruct it.

Packet Loss Simulation (Unit: step):
- i.i.d step erasure: Each step is deleted with probability p (p from a specified list), repeated R times with different seeds.
- Optional: burst erasure / truncation (see arguments)

Output:
- Summary CSV table: One row per p (success_rate, avg_received_steps, avg_received_packets, avg_rank, avg_rank_margin)
- p=0.5 failure count table per episode (should be 0)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
class StepPacket:
    """RLNC packets carried in a step (represented by index+bit)."""

    step_id: int
    indices: Tuple[int, ...]
    bits: Tuple[int, ...]

    @property
    def packet_count(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class Episode:
    query_id: str
    steps: Tuple[StepPacket, ...]

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def total_packets(self) -> int:
        return sum(s.packet_count for s in self.steps)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_rlnc_meta(pred_root: Path) -> Path:
    """Find rlnc_meta.json in pred_root or its parent directories."""
    candidates = [
        pred_root / "rlnc_meta.json",
        pred_root.parent / "rlnc_meta.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: search up to 3 levels up
    cur = pred_root
    for _ in range(3):
        c = cur / "rlnc_meta.json"
        if c.exists():
            return c
        cur = cur.parent
    raise FileNotFoundError(f"Could not find rlnc_meta.json (searching up from {pred_root})")


def iter_episode_files(pred_root: Path) -> List[Path]:
    """Enumerate episode json files (excluding rlnc_meta.json and other non-episode files)."""
    files = []
    for p in sorted(pred_root.rglob("*.json")):
        if p.name == "rlnc_meta.json":
            continue
        # ToolBench watermark prediction files are usually <split>/<query_id>.json
        files.append(p)
    return files


def extract_episode_from_trace(file: Path) -> Episode:
    """Convert watermark_trace to step-level packets (using only index+bit; bits are reconstructed from payload_bits)."""
    data = load_json(file)
    trace = data.get("watermark_trace") or []
    if not trace:
        return Episode(query_id=file.stem, steps=tuple())

    steps: List[StepPacket] = []
    for entry in trace:
        step_id = int(entry.get("round", 0))
        start = entry.get("bit_index_before")
        end = entry.get("bit_index_after")
        if start is None or end is None:
            # No index information, treated as a step without packets
            steps.append(StepPacket(step_id=step_id, indices=tuple(), bits=tuple()))
            continue
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            steps.append(StepPacket(step_id=step_id, indices=tuple(), bits=tuple()))
            continue
        # Note: the actual value of bits comes from the RLNC bitstream, reconstructed later using meta + encoder.get_bit(index)
        indices = tuple(range(start_i, end_i))
        steps.append(StepPacket(step_id=step_id, indices=indices, bits=tuple()))
    return Episode(query_id=file.stem, steps=tuple(steps))


def load_all_episodes(pred_roots: List[Path]) -> List[Episode]:
    """Load and parse all episode JSON files from the given root directories."""
    all_episodes = []
    for root in pred_roots:
        files = iter_episode_files(root)
        for f in files:
            try:
                ep = extract_episode_from_trace(f)
                if ep.query_id != "rlnc_meta":
                    all_episodes.append(ep)
            except Exception as e:
                print(f"[WARN] Failed to load {f}: {e}")
    return all_episodes


def attach_bits_from_encoder(episode: Episode, encoder: DeterministicRLNC) -> Episode:
    """Fill in the bit values for each index in the episode using DeterministicRLNC.get_bit(index)."""
    new_steps: List[StepPacket] = []
    for s in episode.steps:
        if not s.indices:
            new_steps.append(StepPacket(step_id=s.step_id, indices=tuple(), bits=tuple()))
            continue
        bits = tuple(int(encoder.get_bit(i)) for i in s.indices)
        new_steps.append(StepPacket(step_id=s.step_id, indices=s.indices, bits=bits))
    return Episode(query_id=episode.query_id, steps=tuple(new_steps))


def rank_gf2_from_indices(indices: Sequence[int], encoder: DeterministicRLNC) -> int:
    """Construct coefficient matrix and calculate GF(2) rank."""
    if not indices:
        return 0
    # Each row is a length k 0/1 coefficient vector
    matrix = np.array([encoder._generate_coeffs(i) for i in indices], dtype=np.uint8)  # noqa: SLF001
    # Gaussian elimination over GF(2) to calculate rank
    rows, cols = matrix.shape
    rank = 0
    col = 0
    for r in range(rows):
        # Find pivot column
        while col < cols and not matrix[r:, col].any():
            col += 1
        if col >= cols:
            break
        # Find pivot row
        pivot = r + int(np.argmax(matrix[r:, col]))
        if matrix[pivot, col] == 0:
            continue
        if pivot != r:
            matrix[[r, pivot]] = matrix[[pivot, r]]
        # Elimination
        for rr in range(rows):
            if rr != r and matrix[rr, col]:
                matrix[rr] ^= matrix[r]
        rank += 1
        col += 1
    return rank


def simulate_iid_step_erasure(steps: Sequence[StepPacket], p: float, rng: random.Random) -> List[StepPacket]:
    kept = []
    for s in steps:
        if rng.random() >= p:
            kept.append(s)
    return kept


def simulate_burst_step_erasure(steps: Sequence[StepPacket], p: float, burst_len: int, rng: random.Random) -> List[StepPacket]:
    """Randomly delete continuous segments until reaching the target loss ratio (approximate)."""
    n = len(steps)
    if n == 0:
        return []
    target_drop = int(round(n * p))
    if target_drop <= 0:
        return list(steps)
    drop = set()
    while len(drop) < target_drop and len(drop) < n:
        start = rng.randrange(0, n)
        for i in range(start, min(n, start + burst_len)):
            drop.add(i)
            if len(drop) >= target_drop:
                break
    return [s for i, s in enumerate(steps) if i not in drop]


def simulate_truncation(steps: Sequence[StepPacket], keep_ratio: float) -> List[StepPacket]:
    n = len(steps)
    keep_n = int(math.floor(n * keep_ratio))
    return list(steps[:keep_n])


def collect_packets(steps: Sequence[StepPacket]) -> Tuple[List[int], List[int]]:
    indices: List[int] = []
    bits: List[int] = []
    for s in steps:
        indices.extend(list(s.indices))
        bits.extend(list(s.bits))
    return indices, bits


def wilson_95ci(success: int, total: int) -> Tuple[float, float]:
    """Wilson score interval, 95%."""
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


@dataclass
class TrialResult:
    query_id: str
    success: bool
    received_steps: int
    received_packets: int
    rank: int
    packets_per_step: float


def run_trials_for_p(
    episodes: Sequence[Episode],
    encoder: DeterministicRLNC,
    p: float,
    R: int,
    seed_base: int,
    mode: str,
    burst_len: int,
    trunc_keep_ratio: float,
    scope: str,
) -> Tuple[List[TrialResult], Dict[str, int]]:
    """Return results for all trials and failure counts per episode when p=0.5 (empty dict for other p)."""
    results: List[TrialResult] = []
    fail_counter: Dict[str, int] = {}
    if scope == "episode":
        for ep_idx, ep in enumerate(episodes):
            for r in range(R):
                rng = random.Random(seed_base + (ep_idx + 1) * 100000 + r * 1000 + int(p * 1000))
                if mode == "iid":
                    kept_steps = simulate_iid_step_erasure(ep.steps, p, rng)
                elif mode == "burst":
                    kept_steps = simulate_burst_step_erasure(ep.steps, p, burst_len, rng)
                elif mode == "trunc":
                    kept_steps = simulate_truncation(ep.steps, trunc_keep_ratio)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                indices, bits = collect_packets(kept_steps)
                rank = rank_gf2_from_indices(indices, encoder)
                decoded = encoder.decode(indices, bits) if rank >= encoder.n else None  # noqa: SLF001
                success = decoded == "".join(map(str, encoder.payload))  # noqa: SLF001
                pps = (len(indices) / len(kept_steps)) if kept_steps else 0.0
                results.append(
                    TrialResult(
                        query_id=ep.query_id,
                        success=bool(success),
                        received_steps=len(kept_steps),
                        received_packets=len(indices),
                        rank=rank,
                        packets_per_step=pps,
                    )
                )
                if abs(p - 0.5) < 1e-9:
                    if not success:
                        fail_counter[ep.query_id] = fail_counter.get(ep.query_id, 0) + 1
    elif scope == "global":
        # In ToolBench implementation, payload/m is often "globally shared"; a threat model more consistent
        # with this is to treat steps from all episodes as part of one long trajectory: the adversary
        # randomly deletes some steps (erasure), and the remaining packets are collectively used to recover the same m.
        for r in range(R):
            rng = random.Random(seed_base + r * 1000 + int(p * 1000))
            kept_all: List[StepPacket] = []
            for ep_idx, ep in enumerate(episodes):
                # Give each episode an independent but reproducible rng stream
                rng_ep = random.Random(rng.randint(0, 2**31 - 1) + ep_idx * 17)
                if mode == "iid":
                    kept_steps = simulate_iid_step_erasure(ep.steps, p, rng_ep)
                elif mode == "burst":
                    kept_steps = simulate_burst_step_erasure(ep.steps, p, burst_len, rng_ep)
                elif mode == "trunc":
                    kept_steps = simulate_truncation(ep.steps, trunc_keep_ratio)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                kept_all.extend(kept_steps)

            indices, bits = collect_packets(kept_all)
            rank = rank_gf2_from_indices(indices, encoder)
            decoded = encoder.decode(indices, bits) if rank >= encoder.n else None  # noqa: SLF001
            success = decoded == "".join(map(str, encoder.payload))  # noqa: SLF001
            pps = (len(indices) / len(kept_all)) if kept_all else 0.0
            results.append(
                TrialResult(
                    query_id="(global)",
                    success=bool(success),
                    received_steps=len(kept_all),
                    received_packets=len(indices),
                    rank=rank,
                    packets_per_step=pps,
                )
            )
        # In global scope, the concept of "failure count per episode" doesn't apply; return empty
        fail_counter = {}
    else:
        raise ValueError(f"Unknown scope: {scope}")
    return results, fail_counter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Simplified RLNC evaluation config JSON")
    ap.add_argument("--pred_root", default=None, help="Root directory of watermark predictions")
    ap.add_argument("--rlnc_meta", default=None, help="Explicit path to rlnc_meta.json")
    ap.add_argument(
        "--pred_roots",
        nargs="*",
        default=None,
        help="Multiple watermark prediction root directories",
    )
    ap.add_argument("--output_dir", default=None, help="Output directory")
    ap.add_argument("--p_list", default=None, help="Comma-separated list of step erasure probabilities")
    ap.add_argument("--trials", type=int, default=0, help="Number of repetitions R for each p")
    ap.add_argument("--seed", type=int, default=42, help="Random seed baseline")
    ap.add_argument("--mode", choices=["iid", "burst", "trunc"], default=None, help="Erasure model")
    ap.add_argument(
        "--scope",
        choices=["episode", "global"],
        default=None,
        help="Evaluation granularity",
    )
    ap.add_argument("--burst_len", type=int, default=8, help="Continuous loss length L for burst mode")
    ap.add_argument("--trunc_keep_ratio", type=float, default=0.5, help="Prefix retention ratio r for trunc mode")
    ap.add_argument("--min_packets", type=int, default=8, help="Filtering threshold")
    ap.add_argument(
        "--threshold_multipliers",
        default="2,3",
        help="Conditional success rate threshold multipliers",
    )
    args = ap.parse_args()

    # Load Config
    eval_cfg = load_json(Path(args.config)) if args.config else {}
    robust_params = eval_cfg.get("robustness_params", {})

    # 1. Determine Pred Roots
    pred_root_str = args.pred_root or eval_cfg.get("pred_dir")
    pred_roots = [Path(p) for p in (args.pred_roots or ([] if pred_root_str is None else [pred_root_str]))]
    if not pred_roots:
        raise ValueError("Please provide --pred_root, --pred_roots, or a config file with pred_dir")

    # 2. Locate rlnc_meta.json and Payload
    m = ""
    K = 0
    for pr in pred_roots:
        # Priority: 1. CLI --rlnc_meta, 2. Config rlnc_meta_path, 3. Auto-detect
        meta_path = None
        if args.rlnc_meta:
            meta_path = Path(args.rlnc_meta)
        elif eval_cfg.get("rlnc_meta_path"):
            meta_path = Path(eval_cfg["rlnc_meta_path"])
        else:
            try:
                meta_path = find_rlnc_meta(pr)
            except FileNotFoundError:
                meta_path = None
        
        if not meta_path or not meta_path.exists():
            raise FileNotFoundError(f"Could not find rlnc_meta.json for {pr}. Please specify --rlnc_meta.")

        meta = load_json(meta_path)
        payload = meta.get("payload", "")
        key = meta.get("stream_key", 0)
        if not payload:
            raise ValueError(f"rlnc_meta.json is missing payload: {meta_path}")
        if not m:
            m = payload
            K = key
        else:
            if payload != m or key != K:
                raise ValueError(f"Inconsistent rlnc_meta across multiple roots")
    
    encoder = DeterministicRLNC(m, stream_key=K)
    k = encoder.n  # noqa: SLF001

    # 3. Determine Output Dir
    out_dir = Path(args.output_dir) if args.output_dir else (Path(eval_cfg.get("output_dir")) if eval_cfg.get("output_dir") else (pred_roots[0] / "robustness_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Parameters
    p_list_str = args.p_list or robust_params.get("p_list") or "0,0.1,0.2,0.3,0.4,0.5,0.6"
    if isinstance(p_list_str, list):
        p_list = [float(x) for x in p_list_str]
    else:
        p_list = [float(x.strip()) for x in p_list_str.split(",") if x.strip()]
    
    trials = args.trials if args.trials > 0 else robust_params.get("trials", 30)
    mode = args.mode or robust_params.get("mode", "iid")
    scope = args.scope or robust_params.get("scope", "episode")

    episodes = load_all_episodes(pred_roots)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_files: List[Path] = []
    for pr in pred_roots:
        episode_files.extend(iter_episode_files(pr))
    episodes_raw = [extract_episode_from_trace(f) for f in sorted(episode_files)]
    episodes = [attach_bits_from_encoder(ep, encoder) for ep in episodes_raw]

    # Filtering: Must be decodable without loss (rank >= k and packet count >= min_packets)
    eligible: List[Episode] = []
    ineligible: List[Tuple[str, int, int]] = []
    for ep in episodes:
        indices, bits = collect_packets(ep.steps)
        rank = rank_gf2_from_indices(indices, encoder)
        if len(indices) < args.min_packets or rank < k:
            ineligible.append((ep.query_id, len(indices), rank))
        else:
            eligible.append(ep)

    eligible_ratio = (len(eligible) / len(episodes)) if episodes else 0.0

    # per-episode overview (no loss): used to analyze packets/step distribution and filtering ratio
    overview_csv = out_dir / "episode_overview.csv"
    with overview_csv.open("w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["query_id", "total_steps", "total_packets", "packets_per_step", "rank_no_loss", "eligible"])
        eligible_set = {e.query_id for e in eligible}
        for ep in episodes:
            indices, _ = collect_packets(ep.steps)
            rank = rank_gf2_from_indices(indices, encoder)
            pps = (len(indices) / ep.total_steps) if ep.total_steps else 0.0
            wri.writerow([ep.query_id, ep.total_steps, len(indices), f"{pps:.6f}", rank, ep.query_id in eligible_set])

    (out_dir / "ineligible_episodes.json").write_text(
        json.dumps(
            {
                "total_episodes": len(episodes),
                "eligible_episodes": len(eligible),
                "eligible_ratio": eligible_ratio,
                "min_packets": args.min_packets,
                "details": [{"query_id": qid, "packets": pk, "rank": rk} for qid, pk, rk in ineligible],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    threshold_multipliers = [int(x.strip()) for x in args.threshold_multipliers.split(",") if x.strip()]
    thresholds = sorted({max(0, m * k) for m in threshold_multipliers})

    # episode length buckets (short/medium/long): according to the tertiles of total_steps (no loss)
    buckets: Dict[str, List[str]] = {}
    if args.scope == "episode" and eligible:
        lens = sorted([(ep.total_steps, ep.query_id) for ep in eligible])
        n = len(lens)
        t1 = lens[max(0, int(math.floor(n / 3)) - 1)][0]
        t2 = lens[max(0, int(math.floor(2 * n / 3)) - 1)][0]
        buckets = {"short": [], "medium": [], "long": []}
        for length, qid in lens:
            if length <= t1:
                buckets["short"].append(qid)
            elif length <= t2:
                buckets["medium"].append(qid)
            else:
                buckets["long"].append(qid)
        (out_dir / "episode_length_buckets.json").write_text(
            json.dumps(
                {
                    "metric": "total_steps(no_loss)",
                    "thresholds": {"t1": t1, "t2": t2},
                    "counts": {k: len(v) for k, v in buckets.items()},
                    "buckets": buckets,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # Main statistics table
    table_rows: List[Dict[str, object]] = []
    p05_failures: Dict[str, int] = {}
    conditional_rows: List[Dict[str, object]] = []
    bucket_rows: List[Dict[str, object]] = []

    for p in p_list:
        trial_results, fail_counter = run_trials_for_p(
            episodes=eligible,
            encoder=encoder,
            p=p,
            R=trials,
            seed_base=args.seed,
            mode=mode,
            burst_len=args.burst_len,
            trunc_keep_ratio=args.trunc_keep_ratio,
            scope=scope,
        )
        if abs(p - 0.5) < 1e-9:
            p05_failures = fail_counter

        total = len(trial_results)
        success = sum(1 for t in trial_results if t.success)
        lo, hi = wilson_95ci(success, total)

        avg_steps = float(np.mean([t.received_steps for t in trial_results])) if total else 0.0
        avg_packets = float(np.mean([t.received_packets for t in trial_results])) if total else 0.0
        avg_rank = float(np.mean([t.rank for t in trial_results])) if total else 0.0
        avg_margin = avg_rank - k
        avg_pps = float(np.mean([t.packets_per_step for t in trial_results])) if total else 0.0
        p50_pps = float(np.percentile([t.packets_per_step for t in trial_results], 50)) if total else 0.0
        p90_pps = float(np.percentile([t.packets_per_step for t in trial_results], 90)) if total else 0.0

        table_rows.append(
            {
                "p": p,
                "decode_success_rate": success / total if total else 0.0,
                "ci95_low": lo,
                "ci95_high": hi,
                "avg_received_steps": avg_steps,
                "avg_received_packets": avg_packets,
                "avg_packets_per_step": avg_pps,
                "p50_packets_per_step": p50_pps,
                "p90_packets_per_step": p90_pps,
                "avg_rank": avg_rank,
                "avg_rank_margin": avg_margin,
                "trials": total,
            }
        )

        # "Bonus" 1: Conditional success rate (meaningful only in episode scope)
        if args.scope == "episode" and thresholds:
            for thr in thresholds:
                eligible_trials = [t for t in trial_results if t.received_packets >= thr]
                n_thr = len(eligible_trials)
                succ_thr = sum(1 for t in eligible_trials if t.success)
                lo_thr, hi_thr = wilson_95ci(succ_thr, n_thr)
                conditional_rows.append(
                    {
                        "p": p,
                        "threshold_packets": thr,
                        "coverage": (n_thr / total) if total else 0.0,
                        "decode_success_rate": (succ_thr / n_thr) if n_thr else 0.0,
                        "ci95_low": lo_thr,
                        "ci95_high": hi_thr,
                        "trials": n_thr,
                        "all_trials": total,
                    }
                )

        # "Bonus" 2: Success rate vs p for length buckets (short/medium/long)
        if args.scope == "episode" and buckets:
            for bname, qids in buckets.items():
                bucket_trials = [t for t in trial_results if t.query_id in set(qids)]
                nb = len(bucket_trials)
                sb = sum(1 for t in bucket_trials if t.success)
                lo_b, hi_b = wilson_95ci(sb, nb)
                bucket_rows.append(
                    {
                        "p": p,
                        "bucket": bname,
                        "decode_success_rate": (sb / nb) if nb else 0.0,
                        "ci95_low": lo_b,
                        "ci95_high": hi_b,
                        "trials": nb,
                    }
                )

    # Write CSV
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "p",
                "decode_success_rate",
                "ci95_low",
                "ci95_high",
                "avg_received_steps",
                "avg_received_packets",
                "avg_packets_per_step",
                "p50_packets_per_step",
                "p90_packets_per_step",
                "avg_rank",
                "avg_rank_margin",
                "trials",
            ],
        )
        writer.writeheader()
        writer.writerows(table_rows)

    # Write "conditional success rate" CSV (episode scope)
    if args.scope == "episode" and conditional_rows:
        cond_csv = out_dir / "conditional_success.csv"
        with cond_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["p", "threshold_packets", "coverage", "decode_success_rate", "ci95_low", "ci95_high", "trials", "all_trials"],
            )
            writer.writeheader()
            writer.writerows(conditional_rows)

    # Write "length bucket" CSV (episode scope)
    if args.scope == "episode" and bucket_rows:
        buck_csv = out_dir / "bucket_success.csv"
        with buck_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["p", "bucket", "decode_success_rate", "ci95_low", "ci95_high", "trials"])
            writer.writeheader()
            writer.writerows(bucket_rows)

    # p=0.5 failure count per episode (meaningful only in episode scope)
    p05_path = out_dir / "p05_failures.csv"
    with p05_path.open("w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["query_id", "failures_over_R", "R", "note"])
        if scope == "episode":
            for ep in eligible:
                wri.writerow([ep.query_id, p05_failures.get(ep.query_id, 0), trials, ""])
        else:
            wri.writerow(["(global)", "", trials, "Per-episode failures are not counted when scope=global"])

    # Extra: Export p=0.5 per-episode statistics in episode scope (used to explain which episodes are more fragile)
    if args.scope == "episode":
        p05 = 0.5
        stats = {ep.query_id: {"success": 0, "trials": 0, "steps": [], "packets": [], "rank": [], "pps": []} for ep in eligible}
        for ep_idx, ep in enumerate(eligible):
            for r_i in range(trials):
                rng = random.Random(args.seed + (ep_idx + 1) * 100000 + r_i * 1000 + int(p05 * 1000))
                if args.mode == "iid":
                    kept = simulate_iid_step_erasure(ep.steps, p05, rng)
                elif args.mode == "burst":
                    kept = simulate_burst_step_erasure(ep.steps, p05, args.burst_len, rng)
                elif args.mode == "trunc":
                    kept = simulate_truncation(ep.steps, args.trunc_keep_ratio)
                else:
                    raise ValueError(f"Unknown mode: {args.mode}")
                indices, bits = collect_packets(kept)
                rank = rank_gf2_from_indices(indices, encoder)
                decoded = encoder.decode(indices, bits) if rank >= encoder.n else None  # noqa: SLF001
                ok = decoded == "".join(map(str, encoder.payload))  # noqa: SLF001
                s = stats[ep.query_id]
                s["trials"] += 1
                s["success"] += int(bool(ok))
                s["steps"].append(len(kept))
                s["packets"].append(len(indices))
                s["rank"].append(rank)
                s["pps"].append((len(indices) / len(kept)) if kept else 0.0)

        ep_stats_csv = out_dir / "episode_p05_stats.csv"
        with ep_stats_csv.open("w", newline="", encoding="utf-8") as f:
            wri = csv.writer(f)
            wri.writerow(
                [
                    "query_id",
                    "success_rate_p05",
                    "avg_received_steps",
                    "avg_received_packets",
                    "avg_packets_per_step",
                    "p50_packets_per_step",
                    "p90_packets_per_step",
                    "avg_rank",
                    "avg_rank_margin",
                    "trials",
                ]
            )
            for ep in eligible:
                s = stats[ep.query_id]
                trials = s["trials"] or 1
                succ_rate = s["success"] / trials
                avg_steps = float(np.mean(s["steps"])) if s["steps"] else 0.0
                avg_packets = float(np.mean(s["packets"])) if s["packets"] else 0.0
                avg_pps = float(np.mean(s["pps"])) if s["pps"] else 0.0
                p50_pps = float(np.percentile(s["pps"], 50)) if s["pps"] else 0.0
                p90_pps = float(np.percentile(s["pps"], 90)) if s["pps"] else 0.0
                avg_rank = float(np.mean(s["rank"])) if s["rank"] else 0.0
                wri.writerow(
                    [
                        ep.query_id,
                        f"{succ_rate:.6f}",
                        f"{avg_steps:.6f}",
                        f"{avg_packets:.6f}",
                        f"{avg_pps:.6f}",
                        f"{p50_pps:.6f}",
                        f"{p90_pps:.6f}",
                        f"{avg_rank:.6f}",
                        f"{(avg_rank - k):.6f}",
                        trials,
                    ]
                )

    print(f"[OK] saved -> {out_dir}")
    print(f"[OK] summary -> {csv_path}")
    print(f"[OK] p=0.5 failures -> {p05_path}")
    print(f"[OK] episode overview -> {overview_csv}")


if __name__ == "__main__":
    main()
