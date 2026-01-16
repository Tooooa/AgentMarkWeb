"""
可视化模块
职责：为实验结果生成可视化图表

需求：9.1, 9.2, 9.3, 9.4, 9.5
"""

import os
import logging
from typing import Dict, List, Any
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# 已移除中文字体设置，因为所有图表都使用英文


def plot_success_rate_comparison(
    metrics: Dict[str, Any],
    output_path: str,
    title: str = 'ALFWorld Task Success Rate Comparison'
) -> bool:
    """
    实现任务成功率对比图表
    
    使用 matplotlib 创建柱状图，比较基线和水印成功率
    
    Args:
        metrics: 实验指标字典
        output_path: 输出文件路径
        title: 图表标题
    
    Returns:
        success: 图表是否成功生成
    
    需求：9.1
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("未安装 matplotlib，无法生成成功率对比图表")
        return False
    
    logger = logging.getLogger(__name__)
    
    try:
        # 已移除 matplotlib 中文设置
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 准备数据
        categories = ['Baseline\n(No Watermark)', 'Experiment\n(Watermarked)']
        success_rates = [
            metrics['baseline_success_rate'] * 100,
            metrics['watermarked_success_rate'] * 100
        ]
        
        # 创建柱状图
        bars = ax.bar(categories, success_rates, color=['#2ecc71', '#e74c3c'], alpha=0.8, width=0.6)
        
        # 设置标签和标题
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        
        # 在柱子上添加标签
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
        
        # 添加差异注释
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
        
        # 添加网格
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # 保存图形
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"成功率对比图表已保存：{output_path}")
        return True
        
    except Exception as e:
        logger.error(f"生成成功率对比图表时出错：{e}")
        return False


def plot_avg_steps_comparison(
    metrics: Dict[str, Any],
    output_path: str,
    title: str = 'ALFWorld Average Steps Comparison'
) -> bool:
    """
    实现平均步数对比图表
    
    创建柱状图比较平均步数
    
    Args:
        metrics: 实验指标字典
        output_path: 输出文件路径
        title: 图表标题
    
    Returns:
        success: 图表是否成功生成
    
    需求：9.2
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("未安装 matplotlib，无法生成平均步数对比图表")
        return False
    
    logger = logging.getLogger(__name__)
    
    try:
        # 已移除 matplotlib 中文设置
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 准备数据
        categories = ['Baseline\n(No Watermark)', 'Experiment\n(Watermarked)']
        avg_steps = [
            metrics['baseline_avg_steps'],
            metrics['watermarked_avg_steps']
        ]
        
        # 创建柱状图
        bars = ax.bar(categories, avg_steps, color=['#3498db', '#9b59b6'], alpha=0.8, width=0.6)
        
        # 设置标签和标题
        ax.set_ylabel('Average Steps', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 在柱子上添加标签
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
        
        # 添加差异注释
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
        
        # 添加网格
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # 保存图形
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
        # 检查是否存在任务类型数据
        if not metrics.get('metrics_by_task_type'):
            logger.info("无任务类型数据，跳过任务类型对比图表")
            return False
        
        # 已移除 matplotlib 中文设置
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 准备数据
        task_types = list(metrics['metrics_by_task_type'].keys())
        baseline_rates = [
            metrics['metrics_by_task_type'][t]['baseline_success_rate'] * 100
            for t in task_types
        ]
        watermarked_rates = [
            metrics['metrics_by_task_type'][t]['watermarked_success_rate'] * 100
            for t in task_types
        ]
        
        # 设置柱子位置
        x = np.arange(len(task_types))
        width = 0.35
        
        # 创建分组柱状图
        bars1 = ax.bar(x - width/2, baseline_rates, width, 
                      label='Baseline', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width/2, watermarked_rates, width,
                      label='Experiment', color='#e74c3c', alpha=0.8)
        
        # 设置标签和标题
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', '\n') for t in task_types], fontsize=9)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 100)
        
        # 添加网格
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # 保存图形
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
        # 检查是否存在任务类型数据
        if not metrics.get('metrics_by_task_type'):
            logger.info("无任务类型数据，跳过任务类型步数图表")
            return False
        
        # 已移除 matplotlib 中文设置
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 准备数据
        task_types = list(metrics['metrics_by_task_type'].keys())
        baseline_steps = [
            metrics['metrics_by_task_type'][t]['baseline_avg_steps']
            for t in task_types
        ]
        watermarked_steps = [
            metrics['metrics_by_task_type'][t]['watermarked_avg_steps']
            for t in task_types
        ]
        
        # 设置柱子位置
        x = np.arange(len(task_types))
        width = 0.35
        
        # 创建分组柱状图
        bars1 = ax.bar(x - width/2, baseline_steps, width, 
                      label='Baseline', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, watermarked_steps, width,
                      label='Experiment', color='#9b59b6', alpha=0.8)
        
        # 设置标签和标题
        ax.set_ylabel('Average Steps', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', '\n') for t in task_types], fontsize=9)
        ax.legend(fontsize=10)
        
        # 添加网格
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # 保存图形
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
    生成所有可视化图表
    
    Args:
        metrics: 实验指标字典
        output_dir: 输出目录路径
        timestamp: 时间戳（可选，用于文件命名）
    
    Returns:
        chart_paths: 生成的图表路径字典
    
    需求：9.1, 9.2
    """
    logger = logging.getLogger(__name__)
    
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("未安装 matplotlib，跳过所有图表生成")
        return {}
    
    # 生成时间戳
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    chart_paths = {}
    
    # 1. 成功率对比图表
    success_rate_path = os.path.join(output_dir, f'success_rate_comparison_{timestamp}.png')
    if plot_success_rate_comparison(metrics, success_rate_path):
        chart_paths['success_rate_comparison'] = success_rate_path
    
    # 2. 平均步数对比图表
    avg_steps_path = os.path.join(output_dir, f'avg_steps_comparison_{timestamp}.png')
    if plot_avg_steps_comparison(metrics, avg_steps_path):
        chart_paths['avg_steps_comparison'] = avg_steps_path
    
    # 3. 按任务类型的成功率对比
    if metrics.get('metrics_by_task_type'):
        task_type_success_path = os.path.join(output_dir, f'success_rate_by_task_type_{timestamp}.png')
        if plot_success_rate_by_task_type(metrics, task_type_success_path):
            chart_paths['success_rate_by_task_type'] = task_type_success_path
        
        # 4. 按任务类型的平均步数对比
        task_type_steps_path = os.path.join(output_dir, f'avg_steps_by_task_type_{timestamp}.png')
        if plot_steps_by_task_type(metrics, task_type_steps_path):
            chart_paths['avg_steps_by_task_type'] = task_type_steps_path
    
    logger.info(f"生成了 {len(chart_paths)} 个可视化图表")
    
    return chart_paths
