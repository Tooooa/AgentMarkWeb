import os
import json
import argparse
import glob
from statistics import mean, stdev
from tabulate import tabulate


def recursive_find_reports(root_dir: str) -> list:
    pattern = os.path.join(root_dir, "**", "evaluation_report_*.json")
    return glob.glob(pattern, recursive=True)


def analyze(root_dir: str):
    reports = recursive_find_reports(root_dir)
    print(f"Found {len(reports)} report files")

    baseline_stats = {
        'success': [],
        'steps': [],
        'times': []
    }

    watermarked_stats = {
        'success': [],
        'steps': [],
        'times': [],
        'bits_per_task': [],
        'bits_per_step': []
    }

    total_baseline_tasks = 0
    total_watermarked_tasks = 0

    for report_path in reports:
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Baseline
            for res in data.get('baseline_results', []):
                total_baseline_tasks += 1
                baseline_stats['success'].append(1 if res.get('success') else 0)
                if res.get('success'):
                    # Usually compute average steps across all tasks; keep as-is for now.
                    baseline_stats['steps'].append(res.get('num_steps', 50))
                    baseline_stats['times'].append(res.get('time_taken', 0))

            # Watermarked
            for res in data.get('watermarked_results', []):
                total_watermarked_tasks += 1
                watermarked_stats['success'].append(1 if res.get('success') else 0)
                watermarked_stats['steps'].append(res.get('num_steps', 50))
                watermarked_stats['times'].append(res.get('time_taken', 0))

                # Bits stats
                w_stats = res.get('watermark_stats', {})
                bits_embedded = w_stats.get('total_bits_embedded', 0)
                watermarked_stats['bits_per_task'].append(bits_embedded)

                steps = res.get('num_steps', 1)
                if steps > 0:
                    watermarked_stats['bits_per_step'].append(bits_embedded / steps)

        except Exception as e:
            print(f"Failed to read report {report_path}: {e}")

    # Aggregate metrics
    def safe_mean(values):
        return mean(values) if values else 0.0

    sr_c = safe_mean(baseline_stats['success']) * 100
    sr_o = safe_mean(watermarked_stats['success']) * 100

    steps_c = safe_mean(baseline_stats['steps'])
    steps_o = safe_mean(watermarked_stats['steps'])

    time_c = safe_mean(baseline_stats['times'])
    time_o = safe_mean(watermarked_stats['times'])

    bits_task_o = safe_mean(watermarked_stats['bits_per_task'])
    bits_step_o = safe_mean(watermarked_stats['bits_per_step'])

    # Output table
    headers = ["Setting", "SR(C)", "SR(O)", "Steps(C)", "Steps(O)", "Time(C) (s)", "Time(O) (s)", "Bits/task (O)", "Bits/step (O)"]
    row = [
        "OOD (100 tasks)",
        f"{sr_c:.2f}%",
        f"{sr_o:.2f}%",
        f"{steps_c:.2f}",
        f"{steps_o:.2f}",
        f"{time_c:.2f}",
        f"{time_o:.2f}",
        f"{bits_task_o:.2f}",
        f"{bits_step_o:.2f}"
    ]

    print("\n" + "="*80)
    print("Aggregate Analysis Result (Table 4 Style)")
    print("="*80)
    print(tabulate([row], headers=headers, tablefmt="github"))
    print("\n")
    print(f"Total Baseline Tasks: {total_baseline_tasks}")
    print(f"Total Watermarked Tasks: {total_watermarked_tasks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Root output directory containing run reports")
    args = parser.parse_args()

    analyze(args.output_dir)
