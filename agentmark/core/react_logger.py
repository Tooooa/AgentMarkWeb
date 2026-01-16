"""
ReAct 风格的 JSON 轨迹记录器
根据 ReAct 论文格式记录完整的交互历史
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict


class ReActTrajectoryLogger:
    """
    ReAct 风格的轨迹记录器

    轨迹格式改编自 ReAct 论文：
    {
        "task_id": 10,
        "task_type": "pick_and_place_simple",
        "task_description": "put a clean soapbar in toilet",
        "success": true,
        "total_steps": 15,
        "final_reward": 1.0,
        "use_watermark": false,
        "trajectory": [
        
            {
                "step": 1,
                "observation": "You are in the middle of a room...",
                "interaction_history": [...],  // ReAct 风格历史
                "admissible_commands": ["go to cabinet 1", ...],
                "thought": {  // LLM "思考"过程（概率分布）
                    "probabilities": {
                        "go to cabinet 1": 0.25,
                        "go to countertop 1": 0.35,
                        ...
                    },
                    "top_actions": [...]
                },
                "action": "go to countertop 1",
                "reward": 0.0,
                "done": false,
                "watermark_info": {  // 仅水印模式
                    "bits_embedded": 2,
                    "target_behaviors": [...],
                    "context_key": "..."
                }
            },
            ...
        ],
        "metadata": {
            "start_time": "2025-11-17 14:30:00",
            "end_time": "2025-11-17 14:32:15",
            "duration_seconds": 135
        }
    }
    """
    
    def __init__(self, output_dir: str, experiment_name: str = "alfworld_react"):
        """
        初始化记录器

        Args:
            output_dir: 输出目录
            experiment_name: 实验名称
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trajectory_file = os.path.join(
            output_dir, 
            f"alfworld_react_{timestamp}.jsonl"
        )
        self.summary_file = os.path.join(
            output_dir,
            f"summary.txt"
        )
        self.experiment_log_file = os.path.join(
            output_dir,
            f"experiment_log.json"
        )
        
        # 初始化实验日志
        self.experiment_log = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "tasks": []
        }
    
    def log_task_trajectory(
        self, 
        task_result,
        interaction_history: List[Dict] = None
    ):
        """
        记录单个任务的完整轨迹（ReAct 格式）

        Args:
            task_result: TaskResult 对象
            interaction_history: ReAct 风格的交互历史
        """
        task_log = {
            "task_id": task_result.task_id,
            "task_type": task_result.task_type,
            "success": task_result.success,
            "total_steps": task_result.total_steps,
            "final_reward": task_result.final_reward,
            "use_watermark": task_result.use_watermark,
            "trajectory": []
        }
        
        # 记录步骤详情
        if task_result.trajectory:
            for step_data in task_result.trajectory:
                step_log = {
                    "step": step_data.step_num,
                    "observation": step_data.observation,
                    "interaction_history": self._format_history(
                        interaction_history, 
                        step_data.step_num
                    ),
                    "admissible_commands": step_data.admissible_commands,
                    "thought": {
                        "probabilities": step_data.probabilities,
                        "top_actions": self._get_top_actions(
                            step_data.probabilities, 
                            top_k=5
                        )
                    },
                    "action": step_data.selected_action,
                    "reward": step_data.reward,
                    "done": step_data.done
                }
                
                # 添加水印信息（如果适用）
                if task_result.use_watermark:
                    step_log["watermark_info"] = {
                        "bits_embedded": step_data.num_bits_embedded,
                        "target_behaviors": step_data.target_behavior_list,
                        "context_key": step_data.context_for_key
                    }
                
                task_log["trajectory"].append(step_log)
        
        # 添加元数据
        task_log["metadata"] = {
            "timestamp": datetime.now().isoformat()
        }
        
        # 写入 JSONL（每个任务一个条目，带缩进以提高可读性）
        with open(self.trajectory_file, 'a', encoding='utf-8') as f:
            # 使用 indent=2 提高可读性，同时保持 JSONL 兼容性
            # 在每个条目后添加空行以便于手动阅读
            f.write(json.dumps(task_log, ensure_ascii=False, indent=2) + '\n\n')
        
        # 更新实验日志
        self.experiment_log["tasks"].append({
            "task_id": task_result.task_id,
            "success": task_result.success,
            "steps": task_result.total_steps,
            "reward": task_result.final_reward
        })
    
    def _format_history(
        self, 
        interaction_history: List[Dict], 
        current_step: int
    ) -> List[Dict]:
        """
        为当前步骤上下文格式化交互历史

        Args:
            interaction_history: 完整的交互历史
            current_step: 当前步骤编号

        Returns:
            格式化的历史记录
        """
        if not interaction_history:
            return []
        
        # 仅包含当前步骤之前的历史
        history_before_current = interaction_history[:current_step-1]
        
        formatted = []
        for i, step in enumerate(history_before_current, 1):
            formatted.append({
                "step": i,
                "observation": step.get('observation', ''),
                "action": step.get('action', ''),
                "result": "成功" if step.get('reward', 0) > 0 else "继续"
            })
        
        return formatted
    
    def _get_top_actions(
        self, 
        probabilities: Dict[str, float], 
        top_k: int = 5
    ) -> List[Dict[str, float]]:
        """
        按概率获取前 k 个动作

        Args:
            probabilities: 概率分布
            top_k: 要返回的动作数量

        Returns:
            [{"action": "...", "probability": 0.xx}, ...]
        """
        sorted_items = sorted(
            probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [
            {"action": action, "probability": prob}
            for action, prob in sorted_items[:top_k]
        ]
    
    def save_summary(self):
        """保存实验摘要"""
        # 计算统计信息
        total_tasks = len(self.experiment_log["tasks"])
        successful_tasks = sum(
            1 for t in self.experiment_log["tasks"] if t["success"]
        )
        total_steps = sum(
            t["steps"] for t in self.experiment_log["tasks"]
        )
        avg_steps = total_steps / total_tasks if total_tasks > 0 else 0
        
        # 更新实验日志
        self.experiment_log["end_time"] = datetime.now().isoformat()
        self.experiment_log["summary"] = {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "total_steps": total_steps,
            "average_steps": avg_steps
        }
        
        # 保存 JSON 实验日志
        with open(self.experiment_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, ensure_ascii=False, indent=2)
        
        # 保存文本摘要
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ALFWorld ReAct 实验摘要\n")
            f.write(f"实验名称：{self.experiment_name}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"开始时间：{self.experiment_log['start_time']}\n")
            f.write(f"结束时间：{self.experiment_log['end_time']}\n\n")

            f.write(f"总任务数：{total_tasks}\n")
            f.write(f"成功任务数：{successful_tasks}\n")
            f.write(f"成功率：{successful_tasks / total_tasks * 100:.2f}%\n")
            f.write(f"总步数：{total_steps}\n")
            f.write(f"平均步数：{avg_steps:.2f}\n\n")

            f.write("任务详情：\n")
            for task in self.experiment_log["tasks"]:
                status = "成功" if task["success"] else "失败"
                f.write(
                    f"  {status} 任务 {task['task_id']}："
                    f"{task['steps']} 步，奖励 {task['reward']:.2f}\n"
                )
        
        return self.experiment_log_file, self.summary_file
    
    def get_trajectory_file(self) -> str:
        """返回轨迹文件路径"""
        return self.trajectory_file


def create_react_logger(output_dir: str, experiment_name: str) -> ReActTrajectoryLogger:
    """
    创建 ReAct 风格的轨迹记录器

    Args:
        output_dir: 输出目录
        experiment_name: 实验名称

    Returns:
        ReActTrajectoryLogger 实例
    """
    return ReActTrajectoryLogger(output_dir, experiment_name)
