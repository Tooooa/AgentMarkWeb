"""
Experiment controller module.
Responsibilities: orchestrate baseline and watermarked runs, compute metrics, and generate reports.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3, 6.4, 6.5, 9.1, 9.2, 9.3, 9.4, 9.5
"""

import json
import logging
import os
import copy
from typing import Dict, List, Any
from datetime import datetime

from agentmark.environments.alfworld.adapter import ALFWorldAdapter
from agentmark.environments.alfworld.agent import ALFWorldAgent, TaskResult
from agentmark.environments.alfworld.logger import (
    calculate_experiment_metrics,
    log_experiment_start,
    log_task_result
)


def sanitize_config_for_report(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a redacted config copy for reports/logs (avoid writing secrets to disk)."""
    safe_config = copy.deepcopy(config)
    if isinstance(safe_config, dict) and "api_key" in safe_config:
        safe_config["api_key"] = "***REDACTED***"
    return safe_config


def _save_prompts(step_prompts, config: Dict[str, Any], task_id: int, task_index: int, use_watermark: bool):
    """Save prompts for a single task into a standalone JSON file."""
    experiment_config = config.get('experiment_config', {})
    if not experiment_config.get('save_step_prompts', True):
        return
    round_dir = config.get('experiment_context', {}).get('round_dir')
    if not round_dir:
        return
    sub_dir = 'prompts/watermarked' if use_watermark else 'prompts/baseline'
    output_dir = os.path.join(round_dir, sub_dir)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{task_index:03d}_task_{task_id}.json"
    payload = {
        'task_id': task_id,
        'task_index': task_index,
        'use_watermark': use_watermark,
        'prompts': step_prompts
    }
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_baseline_experiment(
    env_adapter: ALFWorldAdapter,
    agent: ALFWorldAgent,
    task_ids: List[int],
    config: Dict[str, Any],
    log_paths: Dict[str, str],
    progress_callback=None,
    on_task_complete=None,
    quiet: bool = False
) -> List[Dict[str, Any]]:
    """
    Run the baseline experiment (no watermark).

    Iterate through task IDs, run the non-watermarked agent, collect results,
    and record progress/statistics.

    Args:
        env_adapter: ALFWorld environment adapter
        agent: ALFWorld agent controller (must be non-watermarked)
        task_ids: List of task IDs
        config: Configuration dict
        log_paths: Log file paths

    Returns:
        baseline_results: List of task result dicts

    Requirements: 3.1, 3.3
    """
    logger = logging.getLogger(__name__)
    if not quiet:
        logger.info(f"Starting baseline run (no watermark), tasks: {len(task_ids)}")

    # Log experiment start
    alfworld_config = config.get('alfworld_config', {})
    log_experiment_start(
        log_paths['experiment'],
        {
            'num_tasks': len(task_ids),
            'max_steps_per_task': alfworld_config.get('max_steps_per_task', 50),
            'random_seed': alfworld_config.get('random_seed', 2025)
        },
        mode='baseline'
    )

    # Initialize results list
    baseline_results = []

    # Max steps
    max_steps = alfworld_config.get('max_steps_per_task', 50)

    # Iterate tasks
    for i, task_id in enumerate(task_ids):
        if not quiet:
            logger.info(f"Baseline - task {i+1}/{len(task_ids)}: task_id={task_id}")

        try:
            task_start = datetime.now()
            result = agent.run_task(task_id, max_steps=max_steps)
            duration = (datetime.now() - task_start).total_seconds()

            # Convert to dict
            result_dict = {
                'task_id': result.task_id,
                'task_type': result.task_type,
                'success': result.success,
                'total_steps': result.total_steps,
                'final_reward': result.final_reward,
                'use_watermark': result.use_watermark,
                'duration_seconds': duration,
                'step_prompts': result.step_prompts,
                'action_sequence': result.action_sequence,
                'trajectory': [
                    {
                        'step_num': step.step_num,
                        'observation': step.observation,
                        'admissible_commands': step.admissible_commands,
                        'probabilities': step.probabilities,
                        'selected_action': step.selected_action,
                        'reward': step.reward,
                        'done': step.done
                    }
                    for step in result.trajectory
                ] if result.trajectory else None
            }

            # Save prompts separately and remove from result payload
            step_prompts = result_dict.pop('step_prompts', [])
            if step_prompts:
                _save_prompts(step_prompts, config, task_id, i + 1, use_watermark=False)

            baseline_results.append(result_dict)

            # Log task result
            log_task_result(log_paths['experiment'], result_dict)
            if on_task_complete:
                on_task_complete(list(baseline_results))

            # Progress
            if progress_callback:
                progress_callback()
            elif not quiet:
                success_count = sum(1 for r in baseline_results if r['success'])
                logger.info(
                    f"Baseline progress: {i+1}/{len(task_ids)}, "
                    f"success: {success_count}/{i+1} ({success_count/(i+1)*100:.1f}%)"
                )

        except Exception as e:
            logger.error(f"Baseline - task {task_id} failed: {e}")
            # Record failed task
            result_dict = {
                'task_id': task_id,
                'task_type': 'unknown',
                'success': False,
                'total_steps': 0,
                'final_reward': 0.0,
                'use_watermark': False,
                'duration_seconds': 0.0,
                'action_sequence': [],
                'step_prompts': [],
                'trajectory': None,
                'error': str(e)
            }
            baseline_results.append(result_dict)
            log_task_result(log_paths['experiment'], result_dict)
            if progress_callback:
                progress_callback()

    # Compute statistics
    success_count = sum(1 for r in baseline_results if r['success'])
    success_rate = success_count / len(baseline_results) if baseline_results else 0
    avg_steps = sum(r['total_steps'] for r in baseline_results) / len(baseline_results) if baseline_results else 0

    logger.info(
        f"Baseline finished: success_rate={success_rate*100:.2f}%, "
        f"avg_steps={avg_steps:.2f}"
    )

    return baseline_results


def run_watermarked_experiment(
    env_adapter: ALFWorldAdapter,
    agent: ALFWorldAgent,
    task_ids: List[int],
    config: Dict[str, Any],
    log_paths: Dict[str, str],
    progress_callback=None,
    on_task_complete=None,
    quiet: bool = False
) -> List[Dict[str, Any]]:
    """
    Run the watermarked experiment.

    Use the same task IDs as baseline, pass the bit stream to the agent,
    collect results, and record progress/statistics.

    Args:
        env_adapter: ALFWorld environment adapter
        agent: ALFWorld agent controller (must be watermarked)
        task_ids: List of task IDs (same as baseline)
        config: Configuration dict
        log_paths: Log file paths

    Returns:
        watermarked_results: List of task result dicts

    Requirements: 3.2, 3.3
    """
    logger = logging.getLogger(__name__)
    if not quiet:
        logger.info(f"Starting watermarked run, tasks: {len(task_ids)}")

    # Log experiment start
    alfworld_config = config.get('alfworld_config', {})
    watermark_config = config.get('watermark_config', {})
    log_experiment_start(
        log_paths['experiment'],
        {
            'num_tasks': len(task_ids),
            'max_steps_per_task': alfworld_config.get('max_steps_per_task', 50),
            'random_seed': alfworld_config.get('random_seed', 2025),
            'watermark_config': watermark_config
        },
        mode='watermarked'
    )

    # Initialize results list
    watermarked_results = []

    # Max steps
    max_steps = alfworld_config.get('max_steps_per_task', 50)

    # Iterate tasks (same IDs as baseline)
    for i, task_id in enumerate(task_ids):
        if not quiet:
            logger.info(f"Watermarked - task {i+1}/{len(task_ids)}: task_id={task_id}")

        try:
            task_start = datetime.now()
            result = agent.run_task(task_id, max_steps=max_steps)
            duration = (datetime.now() - task_start).total_seconds()

            # Watermark statistics
            watermark_stats = {
                'total_bits_embedded': result.watermark_bits_embedded,
                'detection_trace': result.watermark_detection_trace
            }

            # Convert to dict
            result_dict = {
                'task_id': result.task_id,
                'task_type': result.task_type,
                'success': result.success,
                'total_steps': result.total_steps,
                'final_reward': result.final_reward,
                'use_watermark': result.use_watermark,
                'watermark_stats': watermark_stats,
                'duration_seconds': duration,
                'step_prompts': result.step_prompts,
                'action_sequence': result.action_sequence,
                'trajectory': [
                    {
                        'step_num': step.step_num,
                        'observation': step.observation,
                        'admissible_commands': step.admissible_commands,
                        'probabilities': step.probabilities,
                        'selected_action': step.selected_action,
                        'reward': step.reward,
                        'done': step.done,
                        'num_bits_embedded': step.num_bits_embedded,
                        'target_behavior_list': step.target_behavior_list,
                        'context_for_key': step.context_for_key
                    }
                    for step in result.trajectory
                ] if result.trajectory else None
            }

            # Save prompts separately and remove from result payload
            step_prompts = result_dict.pop('step_prompts', [])
            if step_prompts:
                _save_prompts(step_prompts, config, task_id, i + 1, use_watermark=True)

            watermarked_results.append(result_dict)

            # Log task result
            log_task_result(log_paths['experiment'], result_dict)
            if on_task_complete:
                on_task_complete(list(watermarked_results))

            # Progress
            if progress_callback:
                progress_callback()
            elif not quiet:
                success_count = sum(1 for r in watermarked_results if r['success'])
                total_bits = sum(r['watermark_stats']['total_bits_embedded'] for r in watermarked_results)
                logger.info(
                    f"Watermarked progress: {i+1}/{len(task_ids)}, "
                    f"success: {success_count}/{i+1} ({success_count/(i+1)*100:.1f}%), "
                    f"total_bits={total_bits}"
                )

        except Exception as e:
            logger.error(f"Watermarked - task {task_id} failed: {e}")
            # Record failed task
            result_dict = {
                'task_id': task_id,
                'task_type': 'unknown',
                'success': False,
                'total_steps': 0,
                'final_reward': 0.0,
                'use_watermark': True,
                'watermark_stats': {'total_bits_embedded': 0, 'detection_trace': []},
                'duration_seconds': 0.0,
                'action_sequence': [],
                'step_prompts': [],
                'trajectory': None,
                'error': str(e)
            }
            watermarked_results.append(result_dict)
            log_task_result(log_paths['experiment'], result_dict)
            if progress_callback:
                progress_callback()

    # Compute statistics
    success_count = sum(1 for r in watermarked_results if r['success'])
    success_rate = success_count / len(watermarked_results) if watermarked_results else 0
    avg_steps = sum(r['total_steps'] for r in watermarked_results) / len(watermarked_results) if watermarked_results else 0
    total_bits = sum(r['watermark_stats']['total_bits_embedded'] for r in watermarked_results)

    logger.info(
        f"Watermarked finished: success_rate={success_rate*100:.2f}%, "
        f"avg_steps={avg_steps:.2f}, total_bits={total_bits}"
    )

    return watermarked_results


def calculate_metrics(
    baseline_results: List[Dict[str, Any]],
    watermarked_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute performance metrics.

    Calculates primary metrics (success rate, success rate difference),
    secondary metrics (avg steps, avg steps difference), and step increase rate
    on successful paths. Also groups stats by task type.

    Args:
        baseline_results: Baseline results list
        watermarked_results: Watermarked results list

    Returns:
        metrics: Dict with all metrics

    Requirements: 3.5, 6.1, 6.2, 6.3, 6.4, 6.5
    """
    logger = logging.getLogger(__name__)
    logger.info("Computing experiment metrics")

    # Use helper from alfworld_logger
    metrics = calculate_experiment_metrics(baseline_results, watermarked_results)

    logger.info(
        f"Metrics computed: success_rate_diff={metrics['success_rate_diff']*100:.2f}%, "
        f"avg_steps_diff={metrics['avg_steps_diff']:.2f}"
    )

    return metrics


def generate_report(
    baseline_results: List[Dict[str, Any]],
    watermarked_results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_dir: str,
    config: Dict[str, Any]
):
    """
    Generate evaluation reports.

    Creates the JSON report, summary text, and comparison charts for success rate
    and avg steps, and saves them in the output directory.

    Args:
        baseline_results: Baseline results list
        watermarked_results: Watermarked results list
        metrics: Metrics dict
        output_dir: Output directory
        config: Configuration dict

    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating evaluation report, output_dir: {output_dir}")

    experiment_context = config.get('experiment_context', {})
    timestamp = experiment_context.get('run_timestamp') or datetime.now().strftime("%Y%m%d_%H%M%S")
    round_dir = experiment_context.get('round_dir')
    if round_dir:
        report_dir = round_dir
        os.makedirs(report_dir, exist_ok=True)
    else:
        report_root = os.path.join(output_dir, 'reports')
        os.makedirs(report_root, exist_ok=True)
        report_dir = os.path.join(report_root, timestamp)
        os.makedirs(report_dir, exist_ok=True)

    # === 1. Detailed JSON report (Requirement 9.3) ===
    logger.info("Writing detailed JSON report")

    # Additional stats
    baseline_task_count = len(baseline_results)
    watermarked_task_count = len(watermarked_results)
    baseline_success_count = sum(1 for r in baseline_results if r['success'])
    watermarked_success_count = sum(1 for r in watermarked_results if r['success'])

    # Watermark stats
    total_bits_embedded = sum(
        r.get('watermark_stats', {}).get('total_bits_embedded', 0)
        for r in watermarked_results
    )
    avg_bits_per_task = total_bits_embedded / watermarked_task_count if watermarked_task_count > 0 else 0

    report_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': 'alfworld_agentmark_evaluation',
            'description': 'Impact of AgentMark watermarking on ALFWorld benchmark performance',
            'config': sanitize_config_for_report(config)
        },
        'summary': {
            'baseline': {
                'total_tasks': baseline_task_count,
                'successful_tasks': baseline_success_count,
                'failed_tasks': baseline_task_count - baseline_success_count,
                'success_rate': metrics['baseline_success_rate'],
                'avg_steps': metrics['baseline_avg_steps'],
                'avg_duration': metrics.get('baseline_avg_duration', 0)
            },
            'watermarked': {
                'total_tasks': watermarked_task_count,
                'successful_tasks': watermarked_success_count,
                'failed_tasks': watermarked_task_count - watermarked_success_count,
                'success_rate': metrics['watermarked_success_rate'],
                'avg_steps': metrics['watermarked_avg_steps'],
                'avg_duration': metrics.get('watermarked_avg_duration', 0),
                'total_bits_embedded': total_bits_embedded,
                'avg_bits_per_task': avg_bits_per_task
            },
            'comparison': {
                'success_rate_diff': metrics['success_rate_diff'],
                'success_rate_diff_percentage': metrics['success_rate_diff'] * 100,
                'avg_steps_diff': metrics['avg_steps_diff'],
                'duration_diff': metrics.get('duration_diff', 0),
                'both_success_count': metrics.get('both_success_count', 0),
                'step_increase_rate': metrics.get('step_increase_rate', 0)
            }
        },
        'metrics': metrics,
        'baseline_results': baseline_results,
        'watermarked_results': watermarked_results
    }

    report_path = os.path.join(report_dir, f'evaluation_report_{timestamp}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    logger.info(f"JSON report saved: {report_path}")

    # === 2. Summary text ===
    summary_path = os.path.join(report_dir, f'evaluation_summary_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ALFWorld AgentMark Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tasks: {baseline_task_count}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Key metrics\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Baseline success rate: {metrics['baseline_success_rate']*100:.2f}% "
            f"({baseline_success_count}/{baseline_task_count})\n"
        )
        f.write(
            f"Watermarked success rate: {metrics['watermarked_success_rate']*100:.2f}% "
            f"({watermarked_success_count}/{watermarked_task_count})\n"
        )
        f.write(f"Success rate diff: {metrics['success_rate_diff']*100:+.2f}%\n\n")

        f.write(f"Baseline avg steps: {metrics['baseline_avg_steps']:.2f}\n")
        f.write(f"Watermarked avg steps: {metrics['watermarked_avg_steps']:.2f}\n")
        f.write(f"Avg steps diff: {metrics['avg_steps_diff']:+.2f}\n\n")
        f.write(f"Baseline avg duration: {metrics.get('baseline_avg_duration', 0):.2f} s\n")
        f.write(f"Watermarked avg duration: {metrics.get('watermarked_avg_duration', 0):.2f} s\n")
        f.write(f"Duration diff: {metrics.get('duration_diff', 0):+.2f} s\n\n")

        if metrics.get('both_success_count', 0) > 0:
            f.write(f"Tasks where both succeed: {metrics['both_success_count']}\n")
            f.write(f"Step increase rate (successful paths): {metrics.get('step_increase_rate', 0):+.2f}%\n\n")

        f.write(f"Total embedded bits: {total_bits_embedded}\n")
        f.write(f"Avg embedded bits per task: {avg_bits_per_task:.2f}\n\n")

        if metrics.get('metrics_by_task_type'):
            f.write("-" * 80 + "\n")
            f.write("Metrics by task type\n")
            f.write("-" * 80 + "\n")
            for task_type, type_metrics in metrics['metrics_by_task_type'].items():
                f.write(f"\n{task_type}:\n")
                f.write(
                    f"  Baseline success rate: {type_metrics['baseline_success_rate']*100:.2f}%\n"
                )
                f.write(
                    f"  Watermarked success rate: {type_metrics['watermarked_success_rate']*100:.2f}%\n"
                )
                f.write(
                    f"  Success rate diff: {type_metrics.get('success_rate_diff', 0)*100:+.2f}%\n"
                )
                f.write(f"  Baseline avg steps: {type_metrics['baseline_avg_steps']:.2f}\n")
                f.write(f"  Watermarked avg steps: {type_metrics['watermarked_avg_steps']:.2f}\n")

        f.write("\n" + "=" * 80 + "\n")

    logger.info(f"Summary text saved: {summary_path}")
    logger.info("Evaluation report generation complete")

    return {
        'report_dir': report_dir,
        'report_path': report_path,
        'summary_path': summary_path,
        'timestamp': timestamp
    }
