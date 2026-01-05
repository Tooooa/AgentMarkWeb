"""
Visualization Module
Responsibility: Generate visualization charts for experiment results

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import os
import logging
from typing import Dict, List, Any
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Chinese font setup removed as English is used for all plots.


def plot_success_rate_comparison(
    metrics: Dict[str, Any],
    output_path: str,
    title: str = 'ALFWorld Task Success Rate Comparison'
) -> bool:
    """
    Implement task success rate comparison chart
    
    Use matplotlib to create a bar chart comparing baseline and watermarked success rates.
    
    Args:
        metrics: Experiment metrics dictionary
        output_path: Output file path
        title: Chart title
    
    Returns:
        success: Whether the chart was generated successfully
    
    Requirements: 9.1
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("matplotlib not installed, cannot generate success rate comparison chart")
        return False
    
    logger = logging.getLogger(__name__)
    
    try:
        # matplotlib Chinese setup removed
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        categories = ['Baseline\n(No Watermark)', 'Experiment\n(Watermarked)']
        success_rates = [
            metrics['baseline_success_rate'] * 100,
            metrics['watermarked_success_rate'] * 100
        ]
        
        # Create bar chart
        bars = ax.bar(categories, success_rates, color=['#2ecc71', '#e74c3c'], alpha=0.8, width=0.6)
        
        # Set labels and title
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Add labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 1,
                f'{rate:.1f}%',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        # Add difference annotation
        success_rate_diff = metrics.get('success_rate_diff', 0) * 100
        ax.text(
            0.5, 0.95,
            f'Diff: {success_rate_diff:+.2f}%',
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Success rate comparison chart saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating success rate comparison chart: {e}")
        return False


def plot_avg_steps_comparison(
    metrics: Dict[str, Any],
    output_path: str,
    title: str = 'ALFWorld Average Steps Comparison'
) -> bool:
    """
    Implement average steps comparison chart
    
    Create bar chart comparing average steps.
    
    Args:
        metrics: Experiment metrics dictionary
        output_path: Output file path
        title: Chart title
    
    Returns:
        success: Whether the chart was generated successfully
    
    Requirements: 9.2
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("matplotlib not installed, cannot generate average steps comparison chart")
        return False
    
    logger = logging.getLogger(__name__)
    
    try:
        # matplotlib Chinese setup removed
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        categories = ['Baseline\n(No Watermark)', 'Experiment\n(Watermarked)']
        avg_steps = [
            metrics['baseline_avg_steps'],
            metrics['watermarked_avg_steps']
        ]
        
        # Create bar chart
        bars = ax.bar(categories, avg_steps, color=['#3498db', '#9b59b6'], alpha=0.8, width=0.6)
        
        # Set labels and title
        ax.set_ylabel('Average Steps', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add labels on bars
        for bar, steps in zip(bars, avg_steps):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.5,
                f'{steps:.1f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        # Add difference annotation
        avg_steps_diff = metrics.get('avg_steps_diff', 0)
        ax.text(
            0.5, 0.95,
            f'Diff: {avg_steps_diff:+.2f}',
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        )
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Average steps comparison chart saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating average steps comparison chart: {e}")
        return False


def plot_success_rate_by_task_type(
    metrics: Dict[str, Any],
    output_path: str,
    title: str = 'Success Rate by Task Type'
) -> bool:
    """
    Display success rate comparison group by task type
    
    Args:
        metrics: Experiment metrics dictionary
        output_path: Output file path
        title: Chart title
    
    Returns:
        success: Whether the chart was generated successfully
    
    Requirements: 9.1
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("matplotlib not installed, cannot generate task type comparison chart")
        return False
    
    logger = logging.getLogger(__name__)
    
    try:
        # Check if task type data exists
        if not metrics.get('metrics_by_task_type'):
            logger.info("No task type data, skipping task type comparison chart")
            return False
        
        # matplotlib Chinese setup removed
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        task_types = list(metrics['metrics_by_task_type'].keys())
        baseline_rates = [
            metrics['metrics_by_task_type'][t]['baseline_success_rate'] * 100
            for t in task_types
        ]
        watermarked_rates = [
            metrics['metrics_by_task_type'][t]['watermarked_success_rate'] * 100
            for t in task_types
        ]
        
        # Set bars position
        x = np.arange(len(task_types))
        width = 0.35
        
        # Create grouped bar chart
        bars1 = ax.bar(x - width/2, baseline_rates, width, 
                      label='Baseline', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width/2, watermarked_rates, width,
                      label='Experiment', color='#e74c3c', alpha=0.8)
        
        # Set labels and title
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', '\n') for t in task_types], fontsize=9)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 100)
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Task type comparison chart saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating task type comparison chart: {e}")
        return False


def plot_steps_by_task_type(
    metrics: Dict[str, Any],
    output_path: str,
    title: str = 'Average Steps by Task Type'
) -> bool:
    """
    Display average steps comparison group by task type
    
    Args:
        metrics: Experiment metrics dictionary
        output_path: Output file path
        title: Chart title
    
    Returns:
        success: Whether the chart was generated successfully
    
    Requirements: 9.2
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("matplotlib not installed, cannot generate task type steps chart")
        return False
    
    logger = logging.getLogger(__name__)
    
    try:
        # Check if task type data exists
        if not metrics.get('metrics_by_task_type'):
            logger.info("No task type data, skipping task type steps chart")
            return False
        
        # matplotlib Chinese setup removed
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        task_types = list(metrics['metrics_by_task_type'].keys())
        baseline_steps = [
            metrics['metrics_by_task_type'][t]['baseline_avg_steps']
            for t in task_types
        ]
        watermarked_steps = [
            metrics['metrics_by_task_type'][t]['watermarked_avg_steps']
            for t in task_types
        ]
        
        # Set bars position
        x = np.arange(len(task_types))
        width = 0.35
        
        # Create grouped bar chart
        bars1 = ax.bar(x - width/2, baseline_steps, width, 
                      label='Baseline', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, watermarked_steps, width,
                      label='Experiment', color='#9b59b6', alpha=0.8)
        
        # Set labels and title
        ax.set_ylabel('Average Steps', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', '\n') for t in task_types], fontsize=9)
        ax.legend(fontsize=10)
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Task type steps chart saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating task type steps chart: {e}")
        return False


def generate_all_visualizations(
    metrics: Dict[str, Any],
    output_dir: str,
    timestamp: str = None
) -> Dict[str, str]:
    """
    Generate all visualization charts
    
    Args:
        metrics: Experiment metrics dictionary
        output_dir: Output directory path
        timestamp: Timestamp (optional, for file naming)
    
    Returns:
        chart_paths: Dictionary of generated chart paths
    
    Requirements: 9.1, 9.2
    """
    logger = logging.getLogger(__name__)
    
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not installed, skipping all chart generation")
        return {}
    
    # Generate timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    chart_paths = {}
    
    # 1. Success rate comparison chart
    success_rate_path = os.path.join(output_dir, f'success_rate_comparison_{timestamp}.png')
    if plot_success_rate_comparison(metrics, success_rate_path):
        chart_paths['success_rate_comparison'] = success_rate_path
    
    # 2. Average steps comparison chart
    avg_steps_path = os.path.join(output_dir, f'avg_steps_comparison_{timestamp}.png')
    if plot_avg_steps_comparison(metrics, avg_steps_path):
        chart_paths['avg_steps_comparison'] = avg_steps_path
    
    # 3. Success rate comparison by task type
    if metrics.get('metrics_by_task_type'):
        task_type_success_path = os.path.join(output_dir, f'success_rate_by_task_type_{timestamp}.png')
        if plot_success_rate_by_task_type(metrics, task_type_success_path):
            chart_paths['success_rate_by_task_type'] = task_type_success_path
        
        # 4. Average steps comparison by task type
        task_type_steps_path = os.path.join(output_dir, f'avg_steps_by_task_type_{timestamp}.png')
        if plot_steps_by_task_type(metrics, task_type_steps_path):
            chart_paths['avg_steps_by_task_type'] = task_type_steps_path
    
    logger.info(f"Generated {len(chart_paths)} visualization charts")
    
    return chart_paths
