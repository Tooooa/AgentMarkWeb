"""
ALFWorld agent controller module.
Responsibilities: implement the agent control loop for perceive-think-decide-act.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7
"""

import logging
import time
import re
import hashlib
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .adapter import ALFWorldAdapter
from .prompt import generate_alfworld_probability_prompt, generate_alfworld_action_prompt
from ...core.parser_utils import extract_and_normalize_probabilities
from ...core.watermark_sampler import sample_behavior, sample_behavior_differential
from .action_executor import ActionExecutor


@dataclass
class StepData:
    """Single-step execution data."""
    step_num: int
    observation: str
    admissible_commands: List[str]
    probabilities: Dict[str, float]
    selected_action: str
    reward: float
    done: bool
    prompt: str = ""
    llm_response: str = "" # Capture the raw thought/response from LLM
    # Watermark fields (watermarked only)
    num_bits_embedded: int = 0
    target_behavior_list: List[str] = field(default_factory=list)
    context_for_key: str = ""


@dataclass
class TaskResult:
    """Execution result for a single task."""
    task_id: int
    task_type: str
    success: bool
    total_steps: int
    final_reward: float
    use_watermark: bool
    trajectory: Optional[List[StepData]] = None
    watermark_bits_embedded: int = 0
    action_sequence: List[str] = field(default_factory=list)
    watermark_detection_trace: List[Dict[str, Any]] = field(default_factory=list)
    step_prompts: List[Dict[str, Any]] = field(default_factory=list)


class ALFWorldAgent:
    """
    ALFWorld agent controller.

    Implements the perceive-think-decide-act loop, with/without watermark mode.

    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7
    """
    
    # Task-type adaptive max steps
    TASK_TYPE_MAX_STEPS = {
        'pick_and_place_simple': 15,
        'pick_clean_then_place_in_recep': 30,  # reduced from 50 to 30
        'pick_two_obj_and_place': 35,
        'pick_heat_then_place_in_recep': 35,
        'pick_cool_then_place_in_recep': 35,
        'look_at_obj_in_light': 25,
        'pick_and_place_with_movable_recep': 40,
    }
    
    def __init__(
        self,
        client,
        config: Dict,
        env_adapter: ALFWorldAdapter,
        use_watermark: bool = False,
        bit_stream: str = None
    ):
        """
        Initialize agent.

        Args:
            client: OpenAI client
            config: Config dict, includes:
                - model: Model name
                - prompt_config: Prompt config
                - watermark_config: Watermark config (if enabled)
            env_adapter: ALFWorld environment adapter
            use_watermark: Whether to use watermarking
            bit_stream: Watermark bit stream (required if use_watermark=True)

        Requirements: 2.1
        """
        self.logger = logging.getLogger(__name__)
        self.client = client
        self.config = config
        self.env_adapter = env_adapter
        self.use_watermark = use_watermark
        self.bit_stream = bit_stream
        
        # Pull parameters from config
        self.model = config.get('model', 'deepseek-chat')
        self.prompt_config = config.get('prompt_config', {})
        self.watermark_config = config.get('watermark_config', {})
        self._last_prompt: str = ""
        
        # Sampling strategy config
        alfworld_config = config.get('alfworld_config', {})
        self.sampling_strategy = alfworld_config.get('sampling_strategy', 'greedy')  # default greedy
        self.sampling_temperature = alfworld_config.get('sampling_temperature', 1.0)
        
        # Context history (for key generation)
        self.action_history = []
        self.context_window_size = alfworld_config.get('context_window_size', 3)
        
        # ReAct-style interaction history
        self.interaction_history = []  # Keep full history (no cap)
        
        # Bit stream index (for watermark embedding)
        self.bit_index = 0
        
        # Current task state
        self.current_observation = None
        self.current_commands = None
        self.holding_item = None  # Current held item (e.g., "ladle 1"); None means empty
        self._last_llm_response = ""  # Store last LLM response for logging
        self.current_task_description: Optional[str] = None
        self.current_task_type: Optional[str] = None
        self.task_pattern = re.compile(r"Your task is to:\s*(.+)", re.IGNORECASE)
        self.processed_item_status: Dict[str, str] = {}

        executor_config = config.get('alfworld_config', {}).get('action_executor', {})
        self.action_executor = ActionExecutor(
            env_adapter=self.env_adapter,
            navigation_config=executor_config
        )
        
        self.logger.info(
            f"ALFWorld Agent initialized: "
            f"model={self.model}, use_watermark={use_watermark}, "
            f"bit_stream_length={len(bit_stream) if bit_stream else 0}"
        )
    
    def _perceive(self) -> Tuple[str, List[str]]:
        """
        Perceive: get current observation and admissible commands.

        Returns:
            observation: Current observation
            admissible_commands: Admissible commands

        Requirements: 2.1
        """
        try:
            # If first step, use cached observation/commands.
            # Otherwise, environment state was updated in _execute().
            if self.current_observation is None or self.current_commands is None:
                raise RuntimeError("Environment not initialized; call reset() first")
            
            observation = self.current_observation
            commands = self.current_commands
            
            self.logger.debug(
                f"Perception complete: observation_length={len(observation)}, "
                f"num_commands={len(commands)}"
            )
            
            return observation, commands
            
        except Exception as e:
            self.logger.error(f"Perception error: {e}")
            raise RuntimeError(f"Perception failed: {e}")
    
    def _think(self, observation: str, commands: List[str]) -> Dict[str, float]:
        """
        Think: call the LLM to produce an action probability distribution.

        Args:
            observation: Current observation
            commands: Admissible commands

        Returns:
            probabilities: Normalized probability distribution

        Requirements: 2.2, 4.1, 8.2
        """
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Build prompt
                use_few_shot = self.prompt_config.get('use_few_shot', True)
                num_few_shot = self.prompt_config.get('few_shot_count', 2)
                
                task_description = self.current_task_description
                if not task_description:
                    task_description = self._extract_task_from_observation(observation)
                    if task_description:
                        self.current_task_description = task_description
                
                prompt = generate_alfworld_probability_prompt(
                    observation=observation,
                    admissible_commands=commands,
                    task_description=task_description,
                    num_few_shot=num_few_shot if use_few_shot else 0,
                    interaction_history=self.interaction_history,  # pass history
                    holding_item=self.holding_item,  # pass holding state
                    processed_item_status=self.processed_item_status,
                    include_reasoning=self.prompt_config.get('include_reasoning', False)
                )
                self._last_prompt = prompt
                
                self.logger.debug(f"Prompt generated (attempt {attempt + 1}/{max_retries})")
                
                # Call LLM API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful AI assistant that analyzes situations and assigns probabilities to actions."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0  # deterministic output for consistency
                )
                
                response_text = response.choices[0].message.content
                self.logger.debug(f"LLM response length: {len(response_text)}")
                
                # Save last LLM response for external logging
                self._last_llm_response = response_text
                
                # Extract and normalize probabilities
                probabilities = extract_and_normalize_probabilities(
                    response_text,
                    commands,
                    logger=self.logger
                )
                
                # Log probability distribution (top 5)
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                self.logger.info("High-level decision - LLM probability distribution (Top 5):")
                for i, (cmd, prob) in enumerate(sorted_probs[:5], 1):
                    self.logger.info(f"   {i}. {cmd:40s} -> {prob:.4f}")
                self.logger.debug(f"Thinking complete: extracted {len(probabilities)} probabilities")
                
                return probabilities
                
            except Exception as e:
                self.logger.warning(
                    f"Thinking error (attempt {attempt + 1}/{max_retries}): {e}"
                )
                
                if attempt < max_retries - 1:
                    # Wait before retry
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                else:
                    # Final attempt failed, use uniform distribution fallback
                    self.logger.error(
                        "Thinking failed after max retries; using uniform distribution fallback"
                    )
                    uniform_prob = 1.0 / len(commands)
                    return {cmd: uniform_prob for cmd in commands}
    
    def _decide(self, probabilities: Dict[str, float]) -> Tuple[str, List[str], int, str]:
        """
        Decide: select an action based on probabilities (with/without watermark).

        Args:
            probabilities: Action probability distribution

        Returns:
            selected_action: Selected action
            target_behavior_list: Target behaviors (watermark only)
            num_bits_embedded: Embedded bits (watermark only)
            context_for_key: Context key (watermark only)

        Requirements: 2.3, 3.1, 3.2
        """
        try:
            if self.use_watermark:
                # Watermark mode: differential sampling
                if self.bit_stream is None:
                    raise ValueError("Watermark mode requires bit_stream")
                
                # Generate context key
                context_for_key = self._generate_context_for_key()
                
                method = self.watermark_config.get('method', 'differential')
                
                if method == 'green_red':
                    # Red/green list sampling
                    selected_action, target_behavior_list, num_bits_embedded = self._select_action_green_red(
                        probabilities=probabilities,
                        context=context_for_key
                    )
                else:
                    # Default: differential watermark sampling
                    if self.bit_stream is None:
                        raise ValueError("Differential watermark mode requires bit_stream")
                        
                    selected_action, target_behavior_list, num_bits_embedded, _ = sample_behavior_differential(
                        probabilities=probabilities,
                        bit_stream=self.bit_stream,
                        bit_index=self.bit_index,
                        context_for_key=context_for_key,
                        round_num=len(self.action_history)  # use step count as round number
                    )
                
                # Update bit stream index
                self.bit_index += num_bits_embedded
                
                # Log watermark decision
                self.logger.info(
                    f"Watermark decision [{method}] - selected action: '{selected_action}' "
                    f"(embedded {num_bits_embedded} bits, "
                    f"target list size: {len(target_behavior_list)})"
                )
                self.logger.debug(
                    f"Decision complete (watermark): action={selected_action}, "
                    f"bits_embedded={num_bits_embedded}, "
                    f"bit_index={self.bit_index}, "
                    f"context={context_for_key}"
                )
                
                return selected_action, target_behavior_list, num_bits_embedded, context_for_key
                
            else:
                # Baseline mode: use configured sampling strategy
                selected_action = sample_behavior(
                    probabilities=probabilities,
                    seed=None,  # keep randomness
                    round_num=len(self.action_history),
                    strategy=self.sampling_strategy,
                    temperature=self.sampling_temperature
                )
                
                # Log baseline decision
                action_prob = probabilities.get(selected_action, 0.0)
                strategy_info = f"{self.sampling_strategy}"
                if self.sampling_strategy == "temperature":
                    strategy_info += f"(T={self.sampling_temperature})"
                self.logger.info(
                    f"Baseline decision [{strategy_info}] - selected action: '{selected_action}' (prob: {action_prob:.3f})"
                )
                self.logger.debug(f"Decision complete (baseline): action={selected_action}")
                
                return selected_action, [], 0, ""
                
        except Exception as e:
            self.logger.error(f"Decision error: {e}")
            # On failure, randomly select an action
            import random
            selected_action = random.choice(list(probabilities.keys()))
            self.logger.warning(f"Decision failed, random action: {selected_action}")
            return selected_action, [], 0, ""
    
    def _execute(self, action: str) -> Tuple[str, List[str], float, bool]:
        """
        Execute: send the action to the environment.

        Args:
            action: Action to execute

        Returns:
            observation: New observation
            admissible_commands: New admissible commands
            reward: Reward value
            done: Whether task is complete

        Requirements: 2.4
        """
        try:
            # Log execution start
            self.logger.info(f"Execute in environment - send action: '{action}'")
            self.logger.info("   Note: TextWorld decomposes navigation commands into low-level steps.")
            
            # Record observation before execution (for history)
            prev_observation = self.current_observation if self.current_observation else ""
            
            # Execute action
            observation, commands, reward, done = self.action_executor.run(action)

            # Note: history is recorded in step 5 of run_task(); avoid duplicates.
            
            # Update current state
            self.current_observation = observation
            self.current_commands = commands
            
            # Update holding item and processing status
            self._update_holding_item(action, observation)
            self._update_processed_item_status(action)
            
            # Update action history
            self.action_history.append(action)
            
            # Build observation preview (first 100 chars)
            obs_preview = observation[:100].replace('\n', ' ') + ('...' if len(observation) > 100 else '')
            self.logger.info(
                f"Execution complete - reward: {reward}, done: {done}, "
                f"new_commands: {len(commands)}"
            )
            self.logger.info(f"New observation: {obs_preview}")
            self.logger.debug(
                f"Execution complete: action={action}, reward={reward}, "
                f"done={done}, new_commands={len(commands)}"
            )
            
            return observation, commands, reward, done
            
        except Exception as e:
            self.logger.error(f"Execution error: action={action}, error={e}")
            raise RuntimeError(f"Execution failed: {e}")
    
    def _update_holding_item(self, action: str, observation: str):
        """
        Update holding item state.

        Updates self.holding_item based on action and observation:
        - "take X from Y" + "You pick up the X" -> holding X
        - "move X to Y" -> holding None (dropped)
        - "put X in/on Y" -> holding None (dropped)

        Args:
            action: Executed action
            observation: Observation after action
        """
        import re
        
        # Detect pick-up: "take X from Y" + observation contains "You pick up the X"
        if action.startswith("take ") and " from " in action:
            if "You pick up" in observation:
                # Extract item name (e.g., "take ladle 1 from countertop 2")
                match = re.match(r'take\s+(.+?)\s+from\s+', action)
                if match:
                    item_name = match.group(1).strip()
                    self.holding_item = item_name
                    self.logger.info(f"Item state update: now holding '{item_name}'")
                    return
        
        # Detect drop: "move X to Y"
        if action.startswith("move ") and " to " in action:
            if self.holding_item:
                self.logger.info(f"Item state update: dropped '{self.holding_item}', now empty")
            self.holding_item = None
            return

    def _update_processed_item_status(self, action: str):
        """
        Update item processing status (clean/heat/cool) based on action.
        """
        patterns = {
            'cleaned': re.compile(r'^clean\s+(.+?)\s+with\s+sinkbasin', re.IGNORECASE),
            'heated': re.compile(r'^heat\s+(.+?)\s+with\s+microwave', re.IGNORECASE),
            'cooled': re.compile(r'^cool\s+(.+?)\s+with\s+fridge', re.IGNORECASE),
        }
        for status, pattern in patterns.items():
            match = pattern.match(action.strip())
            if match:
                item_name = match.group(1).strip()
                self.processed_item_status[item_name] = status
                self.logger.info(f"Process status update: {item_name} -> {status}")
                break
        
        # Detect drop: "put X in/on Y" (fallback in case used)
        if action.startswith("put "):
            if self.holding_item:
                self.logger.info(f"Item state update: dropped '{self.holding_item}', now empty")
            self.holding_item = None
            return
    
    def run_task(self, task_id: int, max_steps: int = None, react_logger=None) -> TaskResult:
        """
        Run a single task.

        Args:
            task_id: Task ID
            max_steps: Max step limit (None uses task-type adaptive value)
            react_logger: ReAct-style logger (optional)

        Returns:
            result: Task result dict

        Requirements: 2.5, 2.6, 2.7, 5.5
        """
        # Reset agent state
        self.reset_for_new_task()
        
        # Record task start time
        task_start_time = datetime.now()
        
        # Get task info
        task_info = self.env_adapter.get_task_description(task_id)
        task_type = task_info['task_type']
        self.set_task_context(task_info)
        
        # Set max_steps adaptively
        if max_steps is None:
            max_steps = self.TASK_TYPE_MAX_STEPS.get(task_type, 50)
        
        self.logger.info(f"Starting task {task_id}: {task_type}")
        self.logger.info(f"Max steps: {max_steps} (task-type adaptive)")

        
        # Reset environment to target task
        try:
            observation, commands = self.env_adapter.reset(task_id)
            self.current_observation = observation
            self.current_commands = commands
            self.update_task_from_observation(observation)
        except Exception as e:
            self.logger.error(f"Task {task_id} reset failed: {e}")
            return TaskResult(
                task_id=task_id,
                task_type=task_type,
                success=False,
                total_steps=0,
                final_reward=0.0,
                use_watermark=self.use_watermark,
                trajectory=[],
                step_prompts=[]
            )
        
        # Initialize trajectory tracking
        trajectory = []
        action_sequence: List[str] = []
        watermark_detection_trace: List[Dict[str, Any]] = []
        done = False
        final_reward = 0.0
        step_num = 0
        
        # Perceive-think-decide-act loop
        while not done and step_num < max_steps:
            step_num += 1
            
            try:
                # 1. Perceive
                observation, commands = self._perceive()
                
                # 2. Think
                probabilities = self._think(observation, commands)
                
                # 3. Decide
                selected_action, target_list, num_bits, context = self._decide(probabilities)
                
                # 4. Execute
                new_observation, new_commands, reward, done = self._execute(selected_action)
                
                # 5. Record step data
                step_data = StepData(
                    step_num=step_num,
                    observation=observation,
                    admissible_commands=commands,
                    probabilities=probabilities,
                    selected_action=selected_action,
                    reward=reward,
                    done=done,
                    prompt=self._last_prompt,
                    llm_response=self._last_llm_response,
                    num_bits_embedded=num_bits,
                    target_behavior_list=target_list,
                    context_for_key=context
                )
                trajectory.append(step_data)
                action_sequence.append(selected_action)
                
                if self.use_watermark:
                    filtered_probs = {
                        cmd: float(prob)
                        for cmd, prob in probabilities.items()
                        if prob > 0
                    }
                    watermark_detection_trace.append({
                        'step_num': step_num,
                        'action': selected_action,
                        'bits_embedded': num_bits,
                        'target_size': len(target_list),
                        'target_behaviors': list(target_list),
                        'probabilities': filtered_probs,
                        'context_for_key': context,
                        'round_num': step_num - 1
                    })
                
                # Update interaction history (for next prompt generation)
                self.interaction_history.append({
                    'observation': observation,
                    'action': selected_action,
                    'reward': reward
                })
                
                # Update final reward
                final_reward = reward
                
                # Log step summary
                self.logger.info(
                    f"{'='*60}\n"
                    f"Step {step_num}/{max_steps} complete - reward: {reward}, continue: {not done}"
                )
                
            except Exception as e:
                self.logger.error(f"Task {task_id} step {step_num} error: {e}")
                # Stop task on error
                done = True
                break
        
        # Determine task success
        success = (final_reward > 0.0)
        
        total_bits_embedded = sum(step.num_bits_embedded for step in trajectory)
        
        # Decide whether to save full trajectory
        save_failed_trajectories = self.config.get('experiment_config', {}).get('save_failed_trajectories', True)
        
        if success:
            # Successful tasks: keep trajectory for verification
            result_trajectory = trajectory
        else:
            # Failed tasks: keep or drop trajectory based on config
            if save_failed_trajectories:
                # Keep brief trajectory (key info only)
                result_trajectory = trajectory
            else:
                result_trajectory = None
        
        result = TaskResult(
            task_id=task_id,
            task_type=task_type,
            success=success,
            total_steps=step_num,
            final_reward=final_reward,
            use_watermark=self.use_watermark,
            trajectory=result_trajectory,
            watermark_bits_embedded=total_bits_embedded,
            action_sequence=action_sequence,
            watermark_detection_trace=watermark_detection_trace if self.use_watermark else [],
            step_prompts=[
                {
                    'step_num': s.step_num,
                    'prompt': s.prompt
                }
                for s in trajectory
            ]
        )
        
        self.logger.info(
            f"Task {task_id} complete: success={success}, "
            f"steps={step_num}, reward={final_reward}"
        )
        
        # Use ReAct logger to record full trajectory
        if react_logger is not None:
            react_logger.log_task_trajectory(
                task_result=result,
                interaction_history=self.interaction_history
            )
        
        return result
    
    def _generate_context_for_key(self) -> str:
        """
        Generate context key string (uses recent actions).

        Returns:
            context_for_key: Context string for key generation
        """
        recent_actions = self.action_history[-self.context_window_size:] if len(self.action_history) > 0 else []
        return "||".join(recent_actions) if recent_actions else ""
        
    def _select_action_green_red(
        self,
        probabilities: Dict[str, float],
        context: str
    ) -> Tuple[str, List[str], int]:
        """
        Red/green list watermark sampling.

        Args:
            probabilities: Action distribution
            context: Context key

        Returns:
            selected_action: Selected action
            green_list: Green list actions (target_behavior_list)
            num_bits: Embedded bits (fixed 0 for red/green watermark; may be treated as 1 bit/step)
        """
        # Get config
        green_red_config = self.watermark_config.get('green_red_config', {})
        gamma = green_red_config.get('gamma', 0.5)
        delta = green_red_config.get('delta', 2.0)  # Logit bias (defaults to 2.0)
        
        green_list = []
        red_list = []
        
        # 1. Split red/green lists
        for action in probabilities.keys():
            # Hash(context + action)
            h = hashlib.sha256(f"{context}||{action}".encode('utf-8')).hexdigest()
            # Convert to [0, 1) float
            val = int(h[:8], 16) / 0xFFFFFFFF
            
            if val < gamma:
                green_list.append(action)
            else:
                red_list.append(action)
        
        # 2. Sampling logic (Softmax Bias / Soft Watermark)
        # P'(x) = P(x) * exp(delta) if green else P(x)
        # Then normalize
        
        unnormalized_probs = {}
        total_weight = 0.0
        
        bias_factor = math.exp(delta)
        
        for action, prob in probabilities.items():
            weight = prob
            if action in green_list:
                weight *= bias_factor
            
            unnormalized_probs[action] = weight
            total_weight += weight
            
        if total_weight == 0:
            # Fallback (should be rare)
            import random
            selected_action = random.choice(list(probabilities.keys()))
        else:
            # Re-normalize and sample
            normalized_probs = {k: v / total_weight for k, v in unnormalized_probs.items()}
            
            # Weighted sampling
            import random
            r = random.random()
            cur = 0.0
            selected_action = list(probabilities.keys())[0]
            
            for action, p in normalized_probs.items():
                cur += p
                if r <= cur:
                    selected_action = action
                    break
        
        # Red/Green typically does not consume bit stream, so bits=0
        return selected_action, green_list, 0
    
    def reset_for_new_task(self):
        """Reset agent state for a new task."""
        self.action_history = []
        self.interaction_history = []  # Clear history
        self.current_observation = None
        self.current_commands = None
        self.holding_item = None  # Reset holding state
        self.processed_item_status = {}
        self._last_prompt = ""
        # Note: bit_index is not reset because the stream is shared across tasks
        if hasattr(self, 'action_executor'):
            self.action_executor.reset()
        self.current_task_description = None
        self.current_task_type = None

    def set_task_context(self, task_info: Dict[str, str]):
        """
        Set task context manually to provide explicit task description in _think.
        Use when driving the agent from external scripts.
        """
        self.current_task_description = task_info.get('task')
        self.current_task_type = task_info.get('task_type')
    
    def _extract_task_from_observation(self, observation: str) -> Optional[str]:
        """Parse task description from observation text."""
        if not observation:
            return None
        match = self.task_pattern.search(observation)
        if match:
            return match.group(1).strip().rstrip('.')
        return None
    
    def update_task_from_observation(self, observation: str) -> Optional[str]:
        """
        Update current task description from latest observation; return it if found.
        """
        task_desc = self._extract_task_from_observation(observation)
        if task_desc:
            self.current_task_description = task_desc
        return task_desc
