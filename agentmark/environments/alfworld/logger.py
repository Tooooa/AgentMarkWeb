"""
ALFWorld Experiment Logger
Role: Extend logging for task-level and step-level experiment records

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 8.1, 8.2, 8.3, 8.4
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional


def initialize_alfworld_log(
    log_dir: str,
    experiment_name: str = "alfworld_experiment",
    timestamp: Optional[str] = None
) -> Dict[str, str]:
    """
    Initialize ALFWorld experiment log files
    
    Each run creates a timestamped subfolder under log_dir with:
    - experiment.log: Main experiment log (task-level)
    - steps.jsonl: Step-level detailed log (JSONL format)
    - errors.log: Error and exception log
    - summary.json: Experiment summary (generated at end)
    
    Args:
        log_dir: Root log directory path
        experiment_name: Experiment name (for file naming)
        timestamp: Run timestamp (optional, defaults to current time)
    
    Returns:
        log_paths: Dictionary containing all log file paths
    
    Requirements: 5.1, 5.2
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamped run folder
    run_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    log_paths = {
        'experiment': os.path.join(run_dir, f"{experiment_name}_{run_timestamp}.log"),
        'steps': os.path.join(run_dir, f"{experiment_name}_steps_{run_timestamp}.jsonl"),
        'errors': os.path.join(run_dir, f"{experiment_name}_errors_{run_timestamp}.log"),
        'summary': os.path.join(run_dir, f"{experiment_name}_summary_{run_timestamp}.json"),
        'run_dir': run_dir,
        'timestamp': run_timestamp
    }
    
    # Initialize main experiment log
    with open(log_paths['experiment'], 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"ALFWorld Experiment Log\n")
        f.write(f"Experiment Name: {experiment_name}\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
    
    # Initialize error log
    with open(log_paths['errors'], 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"ALFWorld Experiment Error Log\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
    
    # Step-level log file (JSONL format) will be created on first write
    
    return log_paths


def log_experiment_start(log_path: str, config: Dict[str, Any], mode: str):
    """
    Log experiment start information
    
    Args:
        log_path: Main experiment log file path
        config: Experiment configuration dictionary
        mode: Experiment mode ("baseline" or "watermarked")
    
    Requirements: 5.1
    """
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Starting {mode.upper()} Experiment\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        
        # Log key configuration
        f.write("Experiment Configuration:\n")
        f.write(f"  - Number of Tasks: {config.get('num_tasks', 'N/A')}\n")
        f.write(f"  - Max Steps: {config.get('max_steps_per_task', 'N/A')}\n")
        f.write(f"  - Random Seed: {config.get('random_seed', 'N/A')}\n")
        if mode == "watermarked":
            f.write(f"  - Watermark Bit Length: {config.get('watermark_config', {}).get('payload_bit_length', 'N/A')}\n")
        f.write("\n")


def log_task_result(log_path: str, task_result: Dict[str, Any]):
    """
    Log the complete result for a single task
    
    Args:
        log_path: Main experiment log file path
        task_result: Task result dictionary containing:
            - task_id: Task ID
            - task_type: Task type
            - success: Whether successful
            - total_steps: Total steps
            - final_reward: Final reward
            - use_watermark: Whether watermark was used
            - trajectory: Complete trajectory (optional, failed tasks always save brief trajectory)
    
    Requirements: 5.4, 5.5
    """
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"Task {task_result['task_id']} ({task_result.get('task_type', 'unknown')}):\n")
        f.write(f"  Status: {'Success' if task_result['success'] else 'Failed'}\n")
        f.write(f"  Total Steps: {task_result['total_steps']}\n")
        f.write(f"  Final Reward: {task_result['final_reward']}\n")
        f.write(f"  Watermarked: {'Yes' if task_result.get('use_watermark', False) else 'No'}\n")
        if task_result.get('duration_seconds') is not None:
            f.write(f"  Duration: {task_result.get('duration_seconds', 0):.2f}s\n")

        action_sequence = task_result.get('action_sequence', [])
        if action_sequence:
            preview_count = min(len(action_sequence), 10)
            preview = " -> ".join(action_sequence[:preview_count])
            f.write(f"  Action Sequence (first {preview_count} steps): {preview}\n")
            if len(action_sequence) > preview_count:
                f.write(f"  ... {len(action_sequence)} steps total\n")
        
        # For watermark mode, log watermark statistics
        if task_result.get('use_watermark', False) and 'watermark_stats' in task_result:
            stats = task_result['watermark_stats']
            f.write(f"  Bits Embedded: {stats.get('total_bits_embedded', 0)}\n")
            
            detection_trace = stats.get('detection_trace') or []
            if detection_trace:
                f.write(f"  Watermark Trace (first {min(len(detection_trace), 5)} steps):\n")
                for entry in detection_trace[:5]:
                    targets = entry.get('target_behaviors') or []
                    preview_text = ", ".join(targets[:5])
                    f.write(
                        f"    - Step {entry.get('step_num')} Action: {entry.get('action')} | "
                        f"Embedded: {entry.get('bits_embedded', 0)} bits | "
                        f"Target Set Size: {entry.get('target_size', 0)}"
                    )
                    if preview_text:
                        f.write(f" (Examples: {preview_text})")
                    f.write("\n")
        
        # For failed tasks, log brief trajectory
        if not task_result['success'] and 'trajectory' in task_result:
            f.write(f"  Failed Trajectory (first 5 steps):\n")
            trajectory = task_result['trajectory'][:5]  # Only log first 5 steps
            for i, step in enumerate(trajectory):
                f.write(f"    Step {i+1}: {step.get('selected_action', 'N/A')} -> Reward: {step.get('reward', 0)}\n")
        
        f.write("\n")


def log_step_data(steps_log_path: str, step_data: Dict[str, Any]):
    """
    Log detailed data for a single step (JSONL format)
    
    Args:
        steps_log_path: Step-level log file path (JSONL format)
        step_data: Single step data dictionary containing:
            - task_id: Task ID
            - step_num: Step number
            - observation: Observation state
            - admissible_commands: List of executable commands
            - probabilities: Probability distribution
            - selected_action: Selected action
            - reward: Reward
            - done: Whether task is complete
            - use_watermark: Whether watermark was used
            - num_bits_embedded: Number of bits embedded (watermark mode)
            - context_for_key: Context key (watermark mode)
    
    Requirements: 5.3, 5.4
    """
    # Create compact log entry (avoid oversized log files)
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'task_id': step_data['task_id'],
        'step_num': step_data['step_num'],
        'observation_length': len(step_data.get('observation', '')),  # Only log length
        'num_commands': len(step_data.get('admissible_commands', [])),
        'selected_action': step_data.get('selected_action', ''),
        'reward': step_data.get('reward', 0),
        'done': step_data.get('done', False),
        'use_watermark': step_data.get('use_watermark', False)
    }
    
    # For watermark mode, log additional information
    if step_data.get('use_watermark', False):
        log_entry['num_bits_embedded'] = step_data.get('num_bits_embedded', 0)
        log_entry['context_for_key'] = step_data.get('context_for_key', '')
    
    # Log probability distribution statistics (not full distribution)
    if 'probabilities' in step_data and step_data['probabilities']:
        probs = step_data['probabilities']
        log_entry['prob_stats'] = {
            'max_prob': max(probs.values()) if probs else 0,
            'min_prob': min(probs.values()) if probs else 0,
            'selected_prob': probs.get(step_data.get('selected_action', ''), 0)
        }
    
    # Append to file in JSONL format
    with open(steps_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')


def log_error(error_log_path: str, error_type: str, error_message: str, 
              context: Optional[Dict[str, Any]] = None):
    """
    Log errors and exceptions
    
    Args:
        error_log_path: Error log file path
        error_type: Error type (e.g., "LLM_API_ERROR", "ENV_ERROR", "PARSING_ERROR")
        error_message: Error message
        context: Error context information (optional)
    
    Requirements: 8.1, 8.2, 8.3, 8.4
    """
    with open(error_log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {error_type}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Error Message: {error_message}\n")
        
        if context:
            f.write(f"\nContext:\n")
            for key, value in context.items():
                # Truncate overly long values
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "... (truncated)"
                f.write(f"  {key}: {value_str}\n")
        
        f.write("\n")


def log_fallback_strategy(error_log_path: str, strategy_type: str, 
                          task_id: int, step_num: int, details: str):
    """
    Log the use of fallback strategies
    
    Args:
        error_log_path: Error log file path
        strategy_type: Strategy type (e.g., "UNIFORM_DISTRIBUTION", "PARTIAL_NORMALIZATION")
        task_id: Task ID
        step_num: Step number
        details: Detailed explanation
    
    Requirements: 8.2
    """
    with open(error_log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fallback Strategy: {strategy_type}\n")
        f.write(f"  Task ID: {task_id}, Step: {step_num}\n")
        f.write(f"  Details: {details}\n\n")


def calculate_experiment_metrics(baseline_results: List[Dict[str, Any]], 
                                 watermarked_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate experiment metrics
    
    Args:
        baseline_results: Baseline (control) group results list
        watermarked_results: Watermarked (experiment) group results list
    
    Returns:
        metrics: Dictionary containing all metrics
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    """
    # Primary metrics
    baseline_success_count = sum(1 for r in baseline_results if r['success'])
    watermarked_success_count = sum(1 for r in watermarked_results if r['success'])
    
    total_tasks = len(baseline_results)
    
    baseline_success_rate = baseline_success_count / total_tasks if total_tasks > 0 else 0
    watermarked_success_rate = watermarked_success_count / total_tasks if total_tasks > 0 else 0
    success_rate_diff = baseline_success_rate - watermarked_success_rate
    
    # Secondary metrics: average steps
    baseline_avg_steps = sum(r['total_steps'] for r in baseline_results) / total_tasks if total_tasks > 0 else 0
    watermarked_avg_steps = sum(r['total_steps'] for r in watermarked_results) / total_tasks if total_tasks > 0 else 0
    avg_steps_diff = watermarked_avg_steps - baseline_avg_steps

    baseline_avg_duration = sum(r.get('duration_seconds', 0) for r in baseline_results) / total_tasks if total_tasks > 0 else 0
    watermarked_avg_duration = sum(r.get('duration_seconds', 0) for r in watermarked_results) / total_tasks if total_tasks > 0 else 0
    duration_diff = watermarked_avg_duration - baseline_avg_duration
    
    # Success path analysis (only for tasks where both succeeded)
    both_success_tasks = []
    for b_result, w_result in zip(baseline_results, watermarked_results):
        if b_result['success'] and w_result['success']:
            both_success_tasks.append({
                'task_id': b_result['task_id'],
                'baseline_steps': b_result['total_steps'],
                'watermarked_steps': w_result['total_steps'],
                'step_increase': w_result['total_steps'] - b_result['total_steps']
            })
    
    both_success_count = len(both_success_tasks)
    
    if both_success_count > 0:
        total_step_increase = sum(t['step_increase'] for t in both_success_tasks)
        total_baseline_steps = sum(t['baseline_steps'] for t in both_success_tasks)
        step_increase_rate = (total_step_increase / total_baseline_steps) * 100 if total_baseline_steps > 0 else 0
    else:
        step_increase_rate = 0
    
    # Watermark statistics (experiment group only)
    total_bits_embedded = 0
    for w_result in watermarked_results:
        if 'watermark_stats' in w_result:
            total_bits_embedded += w_result['watermark_stats'].get('total_bits_embedded', 0)
    
    avg_bits_per_task = total_bits_embedded / total_tasks if total_tasks > 0 else 0
    
    # Group by task type
    metrics_by_task_type = {}
    task_types = set(r.get('task_type', 'unknown') for r in baseline_results)
    
    for task_type in task_types:
        b_type_results = [r for r in baseline_results if r.get('task_type', 'unknown') == task_type]
        w_type_results = [r for r in watermarked_results if r.get('task_type', 'unknown') == task_type]
        
        if b_type_results:
            type_total = len(b_type_results)
            type_b_success = sum(1 for r in b_type_results if r['success'])
            type_w_success = sum(1 for r in w_type_results if r['success'])
            
            type_b_avg_steps = (
                sum(r['total_steps'] for r in b_type_results) / type_total
            ) if type_total > 0 else 0
            
            type_w_avg_steps = (
                sum(r['total_steps'] for r in w_type_results) / len(w_type_results)
            ) if w_type_results else 0
            
            metrics_by_task_type[task_type] = {
                'total_tasks': type_total,
                'baseline_success_rate': type_b_success / type_total,
                'watermarked_success_rate': type_w_success / type_total,
                'success_rate_diff': (type_b_success - type_w_success) / type_total,
                'baseline_avg_steps': type_b_avg_steps,
                'watermarked_avg_steps': type_w_avg_steps,
                'avg_steps_diff': type_w_avg_steps - type_b_avg_steps
            }
    
    # Aggregate all metrics
    metrics = {
        'total_tasks': total_tasks,
        'baseline_success_count': baseline_success_count,
        'watermarked_success_count': watermarked_success_count,
        'baseline_success_rate': baseline_success_rate,
        'watermarked_success_rate': watermarked_success_rate,
        'success_rate_diff': success_rate_diff,
        'baseline_avg_steps': baseline_avg_steps,
        'watermarked_avg_steps': watermarked_avg_steps,
        'avg_steps_diff': avg_steps_diff,
        'baseline_avg_duration': baseline_avg_duration,
        'watermarked_avg_duration': watermarked_avg_duration,
        'duration_diff': duration_diff,
        'both_success_count': both_success_count,
        'both_success_tasks': both_success_tasks,
        'step_increase_rate': step_increase_rate,
        'total_bits_embedded': total_bits_embedded,
        'avg_bits_per_task': avg_bits_per_task,
        'metrics_by_task_type': metrics_by_task_type
    }
    
    return metrics


def log_experiment_summary(log_path: str, summary_path: str, metrics: Dict[str, Any]):
    """
    Log experiment summary
    
    Args:
        log_path: Main experiment log file path
        summary_path: Summary JSON file path
        metrics: Experiment metrics dictionary
    
    Requirements: 5.6, 9.3
    """
    # Write to main log
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Experiment Summary\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Total Tasks: {metrics['total_tasks']}\n\n")
        
        f.write(f"Baseline (No Watermark):\n")
        f.write(f"  Successful Tasks: {metrics['baseline_success_count']}\n")
        f.write(f"  Success Rate: {metrics['baseline_success_rate']*100:.2f}%\n")
        f.write(f"  Avg Steps: {metrics['baseline_avg_steps']:.2f}\n")
        f.write(f"  Avg Duration: {metrics.get('baseline_avg_duration', 0):.2f}s\n\n")
        
        f.write(f"Watermarked:\n")
        f.write(f"  Successful Tasks: {metrics['watermarked_success_count']}\n")
        f.write(f"  Success Rate: {metrics['watermarked_success_rate']*100:.2f}%\n")
        f.write(f"  Avg Steps: {metrics['watermarked_avg_steps']:.2f}\n")
        f.write(f"  Avg Duration: {metrics.get('watermarked_avg_duration', 0):.2f}s\n")
        f.write(f"  Total Bits Embedded: {metrics['total_bits_embedded']}\n")
        f.write(f"  Avg Bits per Task: {metrics['avg_bits_per_task']:.2f}\n\n")
        
        f.write(f"Performance Impact:\n")
        f.write(f"  Success Rate Diff: {metrics['success_rate_diff']*100:.2f}% ")
        f.write(f"({'decrease' if metrics['success_rate_diff'] > 0 else 'increase'})\n")
        f.write(f"  Avg Steps Diff: {metrics['avg_steps_diff']:.2f} ")
        f.write(f"({'increase' if metrics['avg_steps_diff'] > 0 else 'decrease'})\n")
        f.write(f"  Avg Duration Diff: {metrics.get('duration_diff', 0):.2f}s ")
        f.write(f"({'increase' if metrics.get('duration_diff', 0) > 0 else 'decrease'})\n\n")
        
        f.write(f"Tasks Where Both Succeeded:\n")
        f.write(f"  Count: {metrics['both_success_count']}\n")
        f.write(f"  Step Increase Rate: {metrics['step_increase_rate']:.2f}%\n\n")
        
        if metrics['metrics_by_task_type']:
            f.write(f"By Task Type:\n")
            for task_type, type_metrics in metrics['metrics_by_task_type'].items():
                f.write(f"  {task_type}:\n")
                f.write(f"    Task Count: {type_metrics['total_tasks']}\n")
                f.write(f"    Baseline Success Rate: {type_metrics['baseline_success_rate']*100:.2f}%\n")
                f.write(f"    Watermarked Success Rate: {type_metrics['watermarked_success_rate']*100:.2f}%\n")
                f.write(f"    Success Rate Diff: {type_metrics['success_rate_diff']*100:.2f}%\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"Experiment End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")
    
    # Save JSON format summary
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)


def print_experiment_summary(metrics: Dict[str, Any]):
    """
    Print experiment summary to console
    
    Args:
        metrics: Experiment metrics dictionary
    
    Requirements: 9.1, 9.2
    """
    print(f"\n{'='*80}")
    print(f"Experiment Summary")
    print(f"{'='*80}\n")
    
    print(f"Total Tasks: {metrics['total_tasks']}\n")
    
    print(f"Baseline (No Watermark):")
    print(f"  Successful Tasks: {metrics['baseline_success_count']}")
    print(f"  Success Rate: {metrics['baseline_success_rate']*100:.2f}%")
    print(f"  Avg Steps: {metrics['baseline_avg_steps']:.2f}\n")
    
    print(f"Watermarked:")
    print(f"  Successful Tasks: {metrics['watermarked_success_count']}")
    print(f"  Success Rate: {metrics['watermarked_success_rate']*100:.2f}%")
    print(f"  Avg Steps: {metrics['watermarked_avg_steps']:.2f}")
    print(f"  Total Bits Embedded: {metrics['total_bits_embedded']}")
    print(f"  Avg Bits per Task: {metrics['avg_bits_per_task']:.2f}\n")
    
    print(f"Performance Impact:")
    print(f"  Success Rate Diff: {metrics['success_rate_diff']*100:.2f}% ", end='')
    print(f"({'decrease' if metrics['success_rate_diff'] > 0 else 'increase'})")
    print(f"  Avg Steps Diff: {metrics['avg_steps_diff']:.2f} ", end='')
    print(f"({'increase' if metrics['avg_steps_diff'] > 0 else 'decrease'})\n")
    
    print(f"Tasks Where Both Succeeded:")
    print(f"  Count: {metrics['both_success_count']}")
    print(f"  Step Increase Rate: {metrics['step_increase_rate']:.2f}%\n")
    
    print(f"{'='*80}\n")
