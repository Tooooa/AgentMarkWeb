#!/usr/bin/env python3
"""
ALFWorld large-scale experiment analysis.
Generates conference-style result tables.

Metrics:
- Success Rate (%) +/- std
- Avg Steps +/- std
- Green Ratio (watermark detection metric)
"""

import os
import sys
import json
import glob
import argparse
from collections import defaultdict
import numpy as np


def parse_evaluation_report(report_path):
    """Parse a single evaluation_report.json."""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = {
            'baseline': {'success': None, 'steps': None},
            'watermark': {'success': None, 'steps': None, 'green_ratio': None}
        }

        # Baseline
        if 'baseline_results' in data and data['baseline_results']:
            baseline = data['baseline_results']
            if isinstance(baseline, list) and len(baseline) > 0:
                task = baseline[0]
                results['baseline']['success'] = 1 if task.get('success', False) else 0
                results['baseline']['steps'] = task.get('total_steps', 0)
            elif isinstance(baseline, dict):
                results['baseline']['success'] = 1 if baseline.get('success', False) else 0
                results['baseline']['steps'] = baseline.get('total_steps', 0)

        # Watermark
        if 'watermarked_results' in data and data['watermarked_results']:
            wm = data['watermarked_results']
            if isinstance(wm, list) and len(wm) > 0:
                task = wm[0]
                results['watermark']['success'] = 1 if task.get('success', False) else 0
                results['watermark']['steps'] = task.get('total_steps', 0)

                # Green ratio (if available)
                trajectory = task.get('trajectory', [])
                green_count = 0
                total_count = 0
                for step in trajectory:
                    if 'target_behavior_list' in step and 'selected_action' in step:
                        total_count += 1
                        if step['selected_action'] in step['target_behavior_list']:
                            green_count += 1
                if total_count > 0:
                    results['watermark']['green_ratio'] = green_count / total_count
            elif isinstance(wm, dict):
                results['watermark']['success'] = 1 if wm.get('success', False) else 0
                results['watermark']['steps'] = wm.get('total_steps', 0)

        return results
    except Exception as e:
        print(f"Error parsing {report_path}: {e}")
        return None


def collect_results(base_dir, split_name):
    """Collect results under a given split (ID/OOD)."""
    split_dir = os.path.join(base_dir, split_name)
    if not os.path.exists(split_dir):
        print(f"Directory not found: {split_dir}")
        return None

    all_results = defaultdict(lambda: defaultdict(list))

    task_dirs = glob.glob(os.path.join(split_dir, "task_*"))

    for task_dir in task_dirs:
        # Find reports across rounds
        report_pattern = os.path.join(task_dir, "reports", "*", "round_*", "evaluation_report_*.json")
        reports = glob.glob(report_pattern)

        for report_path in reports:
            result = parse_evaluation_report(report_path)
            if result:
                # Baseline
                if result['baseline']['success'] is not None:
                    all_results['baseline']['success'].append(result['baseline']['success'])
                if result['baseline']['steps'] is not None:
                    all_results['baseline']['steps'].append(result['baseline']['steps'])

                # Watermark
                if result['watermark']['success'] is not None:
                    all_results['watermark']['success'].append(result['watermark']['success'])
                if result['watermark']['steps'] is not None:
                    all_results['watermark']['steps'].append(result['watermark']['steps'])
                if result['watermark']['green_ratio'] is not None:
                    all_results['watermark']['green_ratio'].append(result['watermark']['green_ratio'])

    return all_results


def calculate_stats(data_list):
    """Compute mean and std."""
    if not data_list:
        return None, None
    arr = np.array(data_list)
    return np.mean(arr), np.std(arr)


def format_metric(mean_val, std_val, is_percentage=False, decimals=1):
    """Format metric as mean +/- std."""
    if mean_val is None:
        return "N/A"

    if is_percentage:
        return f"{mean_val*100:.{decimals}f} +/- {std_val*100:.{decimals}f}"
    return f"{mean_val:.{decimals}f} +/- {std_val:.{decimals}f}"


def generate_latex_table(id_results, ood_results):
    """Generate LaTeX table."""

    # Compute stats
    stats = {}
    for split_name, results in [('ID', id_results), ('OOD', ood_results)]:
        stats[split_name] = {}
        for group in ['baseline', 'watermark']:
            stats[split_name][group] = {
                'sr': calculate_stats(results[group]['success']),
                'steps': calculate_stats(results[group]['steps']),
            }
            if group == 'watermark':
                stats[split_name][group]['green'] = calculate_stats(results[group].get('green_ratio', []))

    latex = r"""
\begin{table}[t]
\centering
\caption{ALFWorld Experiment Results with Red-Green List Watermarking ($\gamma=0.5$, $\delta=2.0$)}
\label{tab:alfworld_results}
\small
\begin{tabular}{llcc}
\toprule
\textbf{Split} & \textbf{Method} & \textbf{Success Rate (\%)} & \textbf{Avg. Steps} \\
\midrule
"""

    for split in ['ID', 'OOD']:
        # Baseline
        sr_mean, sr_std = stats[split]['baseline']['sr']
        steps_mean, steps_std = stats[split]['baseline']['steps']
        latex += f"{split} & Baseline & {format_metric(sr_mean, sr_std, True)} & {format_metric(steps_mean, steps_std)} \\\n"

        # Watermark
        sr_mean, sr_std = stats[split]['watermark']['sr']
        steps_mean, steps_std = stats[split]['watermark']['steps']
        latex += f" & + Watermark & {format_metric(sr_mean, sr_std, True)} & {format_metric(steps_mean, steps_std)} \\\n"

        if split == 'ID':
            latex += r"\midrule" + "\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex, stats


def generate_markdown_table(id_results, ood_results):
    """Generate Markdown table."""

    # Compute stats
    stats = {}
    for split_name, results in [('ID', id_results), ('OOD', ood_results)]:
        stats[split_name] = {}
        for group in ['baseline', 'watermark']:
            stats[split_name][group] = {
                'sr': calculate_stats(results[group]['success']),
                'steps': calculate_stats(results[group]['steps']),
                'n': len(results[group]['success'])
            }
            if group == 'watermark':
                stats[split_name][group]['green'] = calculate_stats(results[group].get('green_ratio', []))

    md = "## ALFWorld Experiment Results\n\n"
    md += "**Configuration**: Red-Green List Watermarking (gamma=0.5, delta=2.0)\n\n"

    md += "| Split | Method | Success Rate (%) | Avg. Steps | N |\n"
    md += "|-------|--------|------------------|------------|---|\n"

    for split in ['ID', 'OOD']:
        # Baseline
        sr_mean, sr_std = stats[split]['baseline']['sr']
        steps_mean, steps_std = stats[split]['baseline']['steps']
        n = stats[split]['baseline']['n']
        md += f"| {split} | Baseline | {format_metric(sr_mean, sr_std, True)} | {format_metric(steps_mean, steps_std)} | {n} |\n"

        # Watermark
        sr_mean, sr_std = stats[split]['watermark']['sr']
        steps_mean, steps_std = stats[split]['watermark']['steps']
        n = stats[split]['watermark']['n']
        green_mean, green_std = stats[split]['watermark'].get('green', (None, None))
        md += f"| | + Watermark | {format_metric(sr_mean, sr_std, True)} | {format_metric(steps_mean, steps_std)} | {n} |\n"

    # Green ratio stats
    md += "\n### Watermark Detection Metrics\n\n"
    md += "| Split | Green Ratio (%) |\n"
    md += "|-------|----------------|\n"
    for split in ['ID', 'OOD']:
        green_mean, green_std = stats[split]['watermark'].get('green', (None, None))
        if green_mean is not None:
            md += f"| {split} | {format_metric(green_mean, green_std, True)} |\n"
        else:
            md += f"| {split} | N/A |\n"

    return md, stats


def main():
    parser = argparse.ArgumentParser(description="Analyze ALFWorld experiment results")
    parser.add_argument("--dir", type=str, default="output/alfworld_massive_20260102_010730",
                        help="Base output directory")
    parser.add_argument("--format", type=str, choices=["latex", "markdown", "both"], default="both",
                        help="Output format")
    args = parser.parse_args()

    print(f"Analyzing results from: {args.dir}")

    # Collect results
    id_results = collect_results(args.dir, "ID")
    ood_results = collect_results(args.dir, "OOD")

    if id_results is None or ood_results is None:
        print("Failed to collect results")
        return

    print(f"\nCollected samples:")
    print(f"  ID - Baseline: {len(id_results['baseline']['success'])}, Watermark: {len(id_results['watermark']['success'])}")
    print(f"  OOD - Baseline: {len(ood_results['baseline']['success'])}, Watermark: {len(ood_results['watermark']['success'])}")

    # Generate tables
    if args.format in ["markdown", "both"]:
        md, stats = generate_markdown_table(id_results, ood_results)
        print("\n" + "="*60)
        print("MARKDOWN TABLE:")
        print("="*60)
        print(md)

        # Save
        output_path = os.path.join(args.dir, "results_summary.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"\nSaved to: {output_path}")

    if args.format in ["latex", "both"]:
        latex, stats = generate_latex_table(id_results, ood_results)
        print("\n" + "="*60)
        print("LATEX TABLE:")
        print("="*60)
        print(latex)

        # Save
        output_path = os.path.join(args.dir, "results_table.tex")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
