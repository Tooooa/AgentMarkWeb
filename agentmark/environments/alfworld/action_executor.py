"""ActionExecutor module (Option A: reuse ALFWorld Controller).
Responsibilities: execute high-level commands by reusing ALFWorld controller navigation/interaction.

Based on analysis, ALFWorld already includes a full A* navigation system:
- TextWorld: accepts high-level text commands directly
- THOR: controller decomposes navigation commands automatically

This implementation relies on TextWorld to handle navigation decomposition.

Requirements: 1.3, 2.4, 5.5, 8.1
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .adapter import ALFWorldAdapter


class ActionExecutor:
    """
    High-level command executor (Option A: minimal implementation).

    Design principles:
    - TextWorld already handles high-level commands (e.g., "go to fridge 1")
    - The environment decomposes navigation commands into low-level actions
    - We only need to call env_adapter.step()

    Benefits:
    - Simple (~50 lines)
    - Reuses official ALFWorld behavior
    - High reliability
    """

    def __init__(
        self,
        env_adapter: ALFWorldAdapter,
        navigation_config: Optional[Dict] = None,
    ) -> None:
        """
        Initialize ActionExecutor.

        Args:
            env_adapter: ALFWorld environment adapter
            navigation_config: Navigation config (optional, for future use)
        """
        self.logger = logging.getLogger(__name__)
        self.env_adapter = env_adapter
        self.navigation_config = navigation_config or {}
        
        # Config options (for future use)
        self.max_retries = self.navigation_config.get('max_retries', 3)
        self.enable_logging = self.navigation_config.get('enable_logging', True)
        
        self.logger.info("ActionExecutor initialized (Option A: reuse TextWorld Controller)")

    def reset(self) -> None:
        """
        Reset executor state.

        Call at the start of a new task to clear cached state.
        """
        self.logger.debug("ActionExecutor state reset")

    def run(self, high_level_action: str) -> Tuple[str, List[str], float, bool]:
        """
        Execute a high-level command.

        Args:
            high_level_action: High-level command, e.g.:
                - "go to fridge 1" (navigation)
                - "take apple 1 from countertop 1" (manipulation)
                - "open fridge 1" (manipulation)

        Returns:
            observation: Observation after execution
            admissible_commands: New admissible command list
            reward: Reward value
            done: Whether the task is complete

        Core logic:
            TextWorld handles command decomposition:
            - Navigation: internal A* planner produces low-level actions
            - Manipulation: direct single-step execution
        """
        if not isinstance(high_level_action, str) or not high_level_action.strip():
            raise ValueError("ActionExecutor received an invalid action command")

        action = high_level_action.strip()
        
        if self.enable_logging:
            self.logger.debug(f"Execute high-level command: {action}")

        try:
            # Core call: delegate to environment adapter.
            # TextWorld handles:
            # 1. Navigation commands -> internal A* to multi-step actions
            # 2. Manipulation commands -> single-step execution
            observation, commands, reward, done, info = self.env_adapter.step(action)
            
            if self.enable_logging:
                self.logger.debug(
                    f"Command executed: reward={reward}, done={done}, "
                    f"new_commands={len(commands)}"
                )
            
            return observation, commands, reward, done
            
        except Exception as e:
            self.logger.error(f"High-level command failed: action={action}, error={e}")
            raise RuntimeError(f"ActionExecutor failed: {e}")
