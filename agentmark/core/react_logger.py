"""
ReAct-style JSON trajectory logger.
Records full interaction history based on the ReAct paper format.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict


class ReActTrajectoryLogger:
    """
    ReAct-style trajectory logger.

    Trajectory format adapted from the ReAct paper:
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
                "interaction_history": [...],  // ReAct-style history
                "admissible_commands": ["go to cabinet 1", ...],
                "thought": {  // LLM "thought" process (probability distribution)
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
                "watermark_info": {  // watermark mode only
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
        Initialize the logger.

        Args:
            output_dir: Output directory.
            experiment_name: Experiment name.
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        os.makedirs(output_dir, exist_ok=True)
        
        # Build file paths
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
        
        # Initialize experiment log
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
        Log the full trajectory for a single task (ReAct format).

        Args:
            task_result: TaskResult object.
            interaction_history: ReAct-style interaction history.
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
        
        # Record step details
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
                
                # Add watermark info (if applicable)
                if task_result.use_watermark:
                    step_log["watermark_info"] = {
                        "bits_embedded": step_data.num_bits_embedded,
                        "target_behaviors": step_data.target_behavior_list,
                        "context_key": step_data.context_for_key
                    }
                
                task_log["trajectory"].append(step_log)
        
        # Add metadata
        task_log["metadata"] = {
            "timestamp": datetime.now().isoformat()
        }
        
        # Write JSONL (one task per entry with indent for readability)
        with open(self.trajectory_file, 'a', encoding='utf-8') as f:
            # Use indent=2 for readability while keeping JSONL compatibility.
            # Add a blank line after each entry for easier manual reading.
            f.write(json.dumps(task_log, ensure_ascii=False, indent=2) + '\n\n')
        
        # Update experiment log
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
        Format interaction history for the current step context.

        Args:
            interaction_history: Full interaction history.
            current_step: Current step number.

        Returns:
            Formatted history records.
        """
        if not interaction_history:
            return []
        
        # Only include history before the current step
        history_before_current = interaction_history[:current_step-1]
        
        formatted = []
        for i, step in enumerate(history_before_current, 1):
            formatted.append({
                "step": i,
                "observation": step.get('observation', ''),
                "action": step.get('action', ''),
                "result": "Success" if step.get('reward', 0) > 0 else "Continue"
            })
        
        return formatted
    
    def _get_top_actions(
        self, 
        probabilities: Dict[str, float], 
        top_k: int = 5
    ) -> List[Dict[str, float]]:
        """
        Get top-k actions by probability.

        Args:
            probabilities: Probability distribution.
            top_k: Number of actions to return.

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
        """Save the experiment summary."""
        # Calculate statistics
        total_tasks = len(self.experiment_log["tasks"])
        successful_tasks = sum(
            1 for t in self.experiment_log["tasks"] if t["success"]
        )
        total_steps = sum(
            t["steps"] for t in self.experiment_log["tasks"]
        )
        avg_steps = total_steps / total_tasks if total_tasks > 0 else 0
        
        # Update experiment log
        self.experiment_log["end_time"] = datetime.now().isoformat()
        self.experiment_log["summary"] = {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "total_steps": total_steps,
            "average_steps": avg_steps
        }
        
        # Save JSON experiment log
        with open(self.experiment_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, ensure_ascii=False, indent=2)
        
        # Save text summary
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ALFWorld ReAct Experiment Summary\n")
            f.write(f"Experiment Name: {self.experiment_name}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Start Time: {self.experiment_log['start_time']}\n")
            f.write(f"End Time: {self.experiment_log['end_time']}\n\n")

            f.write(f"Total Tasks: {total_tasks}\n")
            f.write(f"Successful Tasks: {successful_tasks}\n")
            f.write(f"Success Rate: {successful_tasks / total_tasks * 100:.2f}%\n")
            f.write(f"Total Steps: {total_steps}\n")
            f.write(f"Average Steps: {avg_steps:.2f}\n\n")

            f.write("Task Details:\n")
            for task in self.experiment_log["tasks"]:
                status = "OK" if task["success"] else "FAIL"
                f.write(
                    f"  {status} Task {task['task_id']}: "
                    f"{task['steps']} steps, reward {task['reward']:.2f}\n"
                )
        
        return self.experiment_log_file, self.summary_file
    
    def get_trajectory_file(self) -> str:
        """Return the trajectory file path."""
        return self.trajectory_file


def create_react_logger(output_dir: str, experiment_name: str) -> ReActTrajectoryLogger:
    """
    Create a ReAct-style trajectory logger.

    Args:
        output_dir: Output directory.
        experiment_name: Experiment name.

    Returns:
        ReActTrajectoryLogger instance.
    """
    return ReActTrajectoryLogger(output_dir, experiment_name)
