#!/usr/bin/env python3
"""
ALFWorld per-task-category analysis (with std and delta tables).
"""

import os
import sys
import json
import glob
from collections import defaultdict
import numpy as np


def parse_evaluation_report(report_path):
    """Parse a single evaluation_report.json."""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = []

        if 'baseline_results' in data and data['baseline_results']:
            for task in data['baseline_results']:
                results.append({
                    'group': 'baseline',
                    'task_type': task.get('task_type', 'unknown'),
                    'success': 1 if task.get('success', False) else 0,
                    'steps': task.get('total_steps', 0)
                })

        if 'watermarked_results' in data and data['watermarked_results']:
            for task in data['watermarked_results']:
                trajectory = task.get('trajectory', [])
                green_count = 0
                total_count = 0
                for step in trajectory:
                    if 'target_behavior_list' in step and 'selected_action' in step:
                        total_count += 1
                        if step['selected_action'] in step['target_behavior_list']:
                            green_count += 1

                green_ratio = green_count / total_count if total_count > 0 else None

                results.append({
                    'group': 'watermark',
                    'task_type': task.get('task_type', 'unknown'),
                    'success': 1 if task.get('success', False) else 0,
                    'steps': task.get('total_steps', 0),
                    'green_ratio': green_ratio
                })

        return results
    except Exception:
        return []


def collect_all_results(base_dir):
    """Collect results across all reports."""
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for split in ['ID', 'OOD']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue

        task_dirs = glob.glob(os.path.join(split_dir, "task_*"))

        for task_dir in task_dirs:
            report_pattern = os.path.join(task_dir, "reports", "*", "round_*", "evaluation_report_*.json")
            reports = glob.glob(report_pattern)

            for report_path in reports:
                results = parse_evaluation_report(report_path)
                for r in results:
                    task_type = r['task_type']
                    group = r['group']

                    all_data[split][task_type][group + '_success'].append(r['success'])
                    all_data[split][task_type][group + '_steps'].append(r['steps'])
                    if r.get('green_ratio') is not None:
                        all_data[split][task_type]['green_ratio'].append(r['green_ratio'])

    return all_data


def fmt(mean_val, std_val):
    """Format mean +/- std."""
    return f"{mean_val:.1f}+/-{std_val:.1f}"


def fmt_pct(mean_val, std_val):
    """Format percentage mean +/- std."""
    return f"{mean_val*100:.1f}+/-{std_val*100:.1f}"


def generate_tables(all_data):
    """Generate markdown tables."""

    md = "# ALFWorld Results by Task Category\n\n"
    md += "**Configuration**: Red-Green List Watermarking (gamma=0.5, delta=2.0)\n\n"

    # Summary data for delta table
    summary = {}

    for split in ['ID', 'OOD']:
        if split not in all_data:
            continue

        md += f"## {split} Split (Detailed)\n\n"
        md += "| Task Type | SR(Base) | SR(RG) | Steps(Base) | Steps(RG) | Green% | N |\n"
        md += "|-----------|----------|--------|-------------|-----------|--------|---|\n"

        split_data = all_data[split]

        all_baseline_sr = []
        all_watermark_sr = []
        all_baseline_steps = []
        all_watermark_steps = []
        all_green = []

        for task_type in sorted(split_data.keys()):
            data = split_data[task_type]

            b_sr = np.array(data['baseline_success'])
            w_sr = np.array(data['watermark_success'])
            b_steps = np.array(data['baseline_steps'])
            w_steps = np.array(data['watermark_steps'])
            green = np.array(data['green_ratio']) if data['green_ratio'] else np.array([])

            all_baseline_sr.extend(b_sr)
            all_watermark_sr.extend(w_sr)
            all_baseline_steps.extend(b_steps)
            all_watermark_steps.extend(w_steps)
            all_green.extend(green)

            n = len(b_sr)
            sr_c = fmt_pct(np.mean(b_sr), np.std(b_sr))
            sr_o = fmt_pct(np.mean(w_sr), np.std(w_sr))
            steps_c = fmt(np.mean(b_steps), np.std(b_steps))
            steps_o = fmt(np.mean(w_steps), np.std(w_steps))
            green_pct = fmt_pct(np.mean(green), np.std(green)) if len(green) > 0 else "N/A"

            short_name = task_type.replace('_', ' ').title()
            if len(short_name) > 22:
                short_name = short_name[:19] + "..."

            md += f"| {short_name} | {sr_c} | {sr_o} | {steps_c} | {steps_o} | {green_pct} | {n} |\n"

        # Overall
        all_baseline_sr = np.array(all_baseline_sr)
        all_watermark_sr = np.array(all_watermark_sr)
        all_baseline_steps = np.array(all_baseline_steps)
        all_watermark_steps = np.array(all_watermark_steps)
        all_green = np.array(all_green)

        sr_c_mean, sr_c_std = np.mean(all_baseline_sr), np.std(all_baseline_sr)
        sr_o_mean, sr_o_std = np.mean(all_watermark_sr), np.std(all_watermark_sr)
        steps_c_mean, steps_c_std = np.mean(all_baseline_steps), np.std(all_baseline_steps)
        steps_o_mean, steps_o_std = np.mean(all_watermark_steps), np.std(all_watermark_steps)
        green_mean, green_std = np.mean(all_green), np.std(all_green)

        md += f"| **Overall** | **{fmt_pct(sr_c_mean, sr_c_std)}** | **{fmt_pct(sr_o_mean, sr_o_std)}** | **{fmt(steps_c_mean, steps_c_std)}** | **{fmt(steps_o_mean, steps_o_std)}** | **{fmt_pct(green_mean, green_std)}** | **{len(all_baseline_sr)}** |\n"
        md += "\n"

        # Save summary
        summary[split] = {
            'sr_c': (sr_c_mean, sr_c_std),
            'sr_o': (sr_o_mean, sr_o_std),
            'steps_c': (steps_c_mean, steps_c_std),
            'steps_o': (steps_o_mean, steps_o_std),
            'green': (green_mean, green_std),
            'n': len(all_baseline_sr)
        }

    # Delta summary table
    md += "---\n\n"
    md += "## Summary: ID vs OOD Performance\n\n"
    md += "| Split | SR(Base) | SR(RG) | Delta SR | Steps(Base) | Steps(RG) | Delta Steps | Green% | N |\n"
    md += "|-------|----------|--------|----------|-------------|-----------|-------------|--------|---|\n"

    for split in ['ID', 'OOD']:
        if split not in summary:
            continue
        s = summary[split]

        sr_c = f"{s['sr_c'][0]*100:.1f}+/-{s['sr_c'][1]*100:.1f}"
        sr_o = f"{s['sr_o'][0]*100:.1f}+/-{s['sr_o'][1]*100:.1f}"
        delta_sr = (s['sr_o'][0] - s['sr_c'][0]) * 100
        delta_sr_str = f"{delta_sr:+.1f}pp"

        steps_c = f"{s['steps_c'][0]:.1f}+/-{s['steps_c'][1]:.1f}"
        steps_o = f"{s['steps_o'][0]:.1f}+/-{s['steps_o'][1]:.1f}"
        delta_steps = s['steps_o'][0] - s['steps_c'][0]
        delta_steps_str = f"{delta_steps:+.1f}"

        green = f"{s['green'][0]*100:.1f}+/-{s['green'][1]*100:.1f}"

        md += f"| {split} | {sr_c} | {sr_o} | {delta_sr_str} | {steps_c} | {steps_o} | {delta_steps_str} | {green} | {s['n']} |\n"

    md += "\n**Note**: Base = Baseline, RG = Red-Green Watermark, Delta SR = SR(RG) - SR(Base)\n"

    return md


def main():
    base_dir = "output/alfworld_massive_20260102_010730"

    if len(sys.argv) > 1:
        base_dir = sys.argv[1]

    print(f"Analyzing: {base_dir}")

    all_data = collect_all_results(base_dir)
    md = generate_tables(all_data)

    print(md)

    # Save
    try:
        output_path = os.path.join(base_dir, "results_by_category.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"\nSaved to: {output_path}")
    except Exception:
        alt_path = "scripts/alfworld/results_by_category.md"
        with open(alt_path, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"\nSaved to: {alt_path}")


if __name__ == "__main__":
    main()
