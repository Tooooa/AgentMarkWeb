"""
ALFWorld environment adapter.
Purpose: wrap ALFWorld interaction APIs and provide a unified interface.
"""

import os
from typing import Tuple, List, Dict, Optional
import logging


# Task type constants (from ALFWorld sources)
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place"
}


class ALFWorldAdapter:
    """ALFWorld environment adapter (based on AlfredTWEnv)."""
    
    @staticmethod
    def filter_observational_commands(commands: List[str]) -> List[str]:
        """
        Filter observational commands and keep actionable ones.

        Filters:
        - look: repeated environment view
        - inventory: repeated inventory view
        - examine: object inspection (usually unnecessary)

        Based on ReAct alfworld.ipynb implementation.

        Args:
            commands: Original command list

        Returns:
            Filtered command list
        """
        filtered = [
            cmd for cmd in commands
            if not cmd.startswith('look')
            and not cmd.startswith('inventory')
            and not cmd.startswith('examine')
        ]
        
        # If empty after filtering, return original list to avoid no commands.
        if len(filtered) == 0:
            return commands
        
        return filtered
    
    def __init__(self, config_path: str, train_eval: str = "eval_in_distribution", 
                 task_types: Optional[List[int]] = None):
        """
        Initialize ALFWorld environment.

        Args:
            config_path: ALFWorld config path (base_config.yaml)
            train_eval: Dataset split ("train", "eval_in_distribution", "eval_out_of_distribution")
            task_types: Task type list (1-6); None means all task types

        Requirements: 1.1
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing ALFWorld environment: config={config_path}, train_eval={train_eval}")
        
        try:
            import yaml
            from alfworld.agents.environment import get_environment
            
            # Resolve config path (supports relative paths)
            if not os.path.isabs(config_path):
                # Relative path: resolve from project root (AgentMark/)
                # __file__ is at .../new_code/modules/alfworld/adapter.py
                # Move up three levels to AgentMark/
                project_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "..")
                )
                config_path = os.path.join(project_root, config_path)
                self.logger.debug(f"Resolved config path: {config_path}")
            
            # Load config
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.env_type = self.config['env']['type']  # 'AlfredTWEnv'
            self.train_eval = train_eval
            
            # Initialize environment
            self.logger.info(f"Creating environment type: {self.env_type}")
            AlfredTWEnv = get_environment(self.env_type)
            self.alfred_env = AlfredTWEnv(self.config, train_eval=train_eval)
            
            # Save game file list (before init_env)
            self.game_files = self.alfred_env.game_files
            self.current_game_idx = 0
            
            # Initialize TextWorld environment
            self.env = self.alfred_env.init_env(batch_size=1)
            
            # Task type filter
            self.task_types_filter = task_types
            if task_types:
                self.logger.info(f"Filter task types: {task_types}")
                self._filter_game_files_by_task_type(task_types)
            
            self.logger.info(f"Environment initialized, available tasks: {len(self.game_files)}")
            
        except ImportError as e:
            self.logger.error(f"Failed to import ALFWorld: {e}")
            raise RuntimeError(f"ALFWorld not installed or import failed: {e}")
        except Exception as e:
            self.logger.error(f"Environment initialization failed: {e}")
            raise RuntimeError(f"ALFWorld environment initialization failed: {e}")
    
    def _filter_game_files_by_task_type(self, task_types: List[int]):
        """
        Filter game files by task types.

        Args:
            task_types: Task types to keep (1-6)
        """
        task_names = [TASK_TYPES[t] for t in task_types if t in TASK_TYPES]
        
        filtered_files = []
        for game_file in self.game_files:
            for task_name in task_names:
                if task_name in game_file:
                    filtered_files.append(game_file)
                    break
        
        original_count = len(self.game_files)
        self.game_files = filtered_files
        self.logger.info(f"Task filter: {original_count} -> {len(self.game_files)}")
    
    def _load_specific_game(self, game_idx: int) -> Tuple[str, List[str]]:
        """
        Load a specific game file by index.

        Args:
            game_idx: Game file index

        Returns:
            observation: Current observation (text)
            admissible_commands: Admissible commands

        Note: This method directly uses the underlying batch_env to load
        a specific game, bypassing TextWorld random selection.
        """
        # Get the target game file
        game_file = self.game_files[game_idx]
        self.logger.debug(f"Loading specific game: idx={game_idx}, file={game_file}")
        
        # Close current environment if present
        if hasattr(self.env, 'batch_env') and self.env.batch_env is not None:
            self.env.batch_env.close()
        
        # Load the game file directly (batch_size=1, so pass a single file list)
        self.env.batch_env.load([game_file])
        
        # Reset environment
        self.env.last_commands = [None] * self.env.batch_size
        obs, info = self.env.batch_env.reset()
        self.env.obs = obs
        
        # Extract observation and commands (batch_size=1, take index 0)
        observation = obs[0]
        admissible_commands = info['admissible_commands'][0]
        
        return observation, admissible_commands
    
    def reset(self, game_idx: Optional[int] = None) -> Tuple[str, List[str]]:
        """
        Reset environment to a new task.

        Args:
            game_idx: Game index (ensure baseline/watermarked use the same task).
                     If None, use current index.

        Returns:
            observation: Current observation (text)
            admissible_commands: Admissible commands

        Requirements: 1.2
        """
        try:
            # Determine game index to load
            if game_idx is not None:
                if game_idx < 0 or game_idx >= len(self.game_files):
                    raise ValueError(f"Invalid game index: {game_idx}, valid range: [0, {len(self.game_files)-1}]")
                self.current_game_idx = game_idx
            
            self.logger.debug(f"Reset environment to task {self.current_game_idx}")
            
            # Load specific game
            observation, admissible_commands = self._load_specific_game(self.current_game_idx)
            
            # Filter observational commands
            original_count = len(admissible_commands)
            admissible_commands = self.filter_observational_commands(admissible_commands)
            
            self.logger.debug(f"Reset ok, commands: {original_count} -> {len(admissible_commands)} (filtered)")
            
            return observation, admissible_commands
            
        except Exception as e:
            self.logger.error(f"Environment reset failed: {e}")
            raise RuntimeError(f"Environment reset failed (game_idx={game_idx}): {e}")
    
    def step(self, action: str) -> Tuple[str, List[str], float, bool, Dict]:
        """
        Execute an action.

        Args:
            action: Command string to execute

        Returns:
            observation: New observation
            admissible_commands: New admissible commands
            reward: Reward (1.0 success, 0.0 otherwise)
            done: Whether the task is finished
            info: Extra info dict

        Requirements: 1.3
        """
        try:
            self.logger.debug(f"Execute action: {action}")
            
            obs, scores, dones, infos = self.env.step([action])  # batch_size=1
            
            observation = obs[0]
            reward = scores[0]
            done = dones[0]
            info = infos
            admissible_commands = info['admissible_commands'][0]
            
            # Filter observational commands
            admissible_commands = self.filter_observational_commands(admissible_commands)
            
            self.logger.debug(
                f"Action complete: reward={reward}, done={done}, commands={len(admissible_commands)} (filtered)"
            )
            
            return observation, admissible_commands, reward, done, info
            
        except Exception as e:
            self.logger.error(f"Action failed: action={action}, error={e}")
            # On invalid action, raise and let the agent try other actions.
            raise RuntimeError(f"Action failed: {e}")
    
    def get_num_games(self) -> int:
        """
        Get number of available games.

        Returns:
            Game count

        Requirements: 1.4, 7.5
        """
        return len(self.game_files)
    
    def get_game_file(self, game_idx: int) -> str:
        """
        Get game file path.

        Args:
            game_idx: Game index

        Returns:
            Game file path

        Requirements: 1.4, 7.5
        """
        if game_idx < 0 or game_idx >= len(self.game_files):
            raise ValueError(f"Invalid game index: {game_idx}, valid range: [0, {len(self.game_files)-1}]")
        return self.game_files[game_idx]
    
    def get_task_info(self, info: Dict) -> Dict[str, str]:
        """
        Extract task info from environment info (see ReAct notebook).

        Args:
            info: info dict returned by environment

        Returns:
            Task info dict:
                - task_name: Full task name
                - task_type: Task type prefix
                - prompt_type: Prompt type

        Requirements: 1.1, 1.2, 1.3, 1.4
        """
        # Extract game file path
        game_file = info['extra.gamefile'][0]
        
        # Extract task name (last 3rd and 2nd path segments)
        # Example: "pick_cool_then_place_in_recep-Bread-None-CounterTop-10/trial_T20190908_091811_414150"
        task_name = '/'.join(game_file.split('/')[-3:-1])
        
        # Match task type (order matters: more specific prefixes first)
        TASK_TYPE_PREFIXES = {
            'pick_clean_then_place': 'clean',  # Must be before pick_and_place
            'pick_heat_then_place': 'heat',
            'pick_cool_then_place': 'cool',
            'pick_two_obj': 'puttwo',
            'pick_and_place': 'put',
            'look_at_obj': 'examine',
        }
        
        for task_prefix, prompt_type in TASK_TYPE_PREFIXES.items():
            if task_name.startswith(task_prefix):
                self.logger.debug(
                    f"Detected task type: task_name={task_name}, task_type={task_prefix}, prompt_type={prompt_type}"
                )
                return {
                    'task_name': task_name,
                    'task_type': task_prefix,
                    'prompt_type': prompt_type
                }
        
        # No match
        self.logger.warning(f"Unrecognized task type: {task_name}")
        return {
            'task_name': task_name,
            'task_type': 'unknown',
            'prompt_type': 'put'  # Default to put type
        }
    
    def get_task_type(self, game_idx: int) -> str:
        """
        Get task type.

        Args:
            game_idx: Game index

        Returns:
            Task type name (e.g., "pick_and_place_simple")

        Requirements: 1.4, 7.5
        """
        game_file = self.get_game_file(game_idx)
        
        # Extract task type from file path
        # Example: .../pick_and_place_simple-...
        for task_name in TASK_TYPES.values():
            if task_name in game_file:
                return task_name
        
        return "unknown"
    
    def get_task_list(self) -> List[int]:
        """
        Get list of all task IDs.

        Returns:
            Task ID list (game index list)

        Requirements: 1.4, 7.5
        """
        return list(range(len(self.game_files)))
    
    def get_task_description(self, game_idx: int) -> Dict[str, str]:
        """
        Get task description.

        Args:
            game_idx: Game index

        Returns:
            Task description dict:
                - task_id: Task ID
                - task_type: Task type
                - game_file: Game file path

        Requirements: 1.4, 7.5
        """
        return {
            "task_id": game_idx,
            "task_type": self.get_task_type(game_idx),
            "game_file": self.get_game_file(game_idx)
        }
    
    def close(self):
        """Close environment and release resources."""
        try:
            if hasattr(self, 'env') and self.env is not None:
                self.logger.info("Closing ALFWorld environment")
                # ALFWorld may not expose close; clear references.
                self.env = None
        except Exception as e:
            self.logger.warning(f"Warning while closing environment: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
