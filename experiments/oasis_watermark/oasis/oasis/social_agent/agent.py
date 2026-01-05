# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import inspect
import logging
import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelManager
from camel.prompts import TextPrompt
from camel.toolkits import FunctionTool
from camel.types import OpenAIBackendRole

from oasis.social_agent.agent_action import SocialAction
from oasis.social_agent.agent_environment import SocialEnvironment
from oasis.social_platform import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType

# Watermark integration
try:
    from oasis.watermark import WatermarkManager
    WATERMARK_AVAILABLE = True
except ImportError:
    WATERMARK_AVAILABLE = False
    WatermarkManager = None

if TYPE_CHECKING:
    from oasis.social_agent import AgentGraph

if "sphinx" not in sys.modules:
    agent_log = logging.getLogger(name="social.agent")
    agent_log.setLevel("DEBUG")

    if not agent_log.handlers:
        log_dir = os.getenv("OASIS_LOG_DIR", "./log")
        os.makedirs(log_dir, exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"social.agent-{str(now)}.log")
        )
        file_handler.setLevel("DEBUG")
        file_handler.setFormatter(
            logging.Formatter(
                "%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
        agent_log.addHandler(file_handler)

ALL_SOCIAL_ACTIONS = [action.value for action in ActionType]


class SocialAgent(ChatAgent):
    r"""Social Agent."""

    def __init__(self,
                 agent_id: int,
                 user_info: UserInfo,
                 user_info_template: TextPrompt | None = None,
                 channel: Channel | None = None,
                 model: Optional[Union[BaseModelBackend,
                                       List[BaseModelBackend],
                                       ModelManager]] = None,
                 agent_graph: "AgentGraph" = None,
                 available_actions: list[ActionType] = None,
                 tools: Optional[List[Union[FunctionTool, Callable]]] = None,
                 max_iteration: int = 1,
                 interview_record: bool = False,
                 watermark_manager: Optional[Any] = None):
        self.social_agent_id = agent_id
        self.user_info = user_info
        self.channel = channel or Channel()
        self.env = SocialEnvironment(SocialAction(agent_id, self.channel))
        
        # Watermark integration - 🎯 自动为每个agent创建独立的水印管理器
        if watermark_manager is None and WATERMARK_AVAILABLE:
            # 自动创建独立的WatermarkManager，使用agent_id作为水印内容
            # 🎯 启用循环嵌入策略
            watermark_log_dir = os.getenv(
                "OASIS_WATERMARK_LOG_DIR",
                os.getenv("OASIS_LOG_DIR", "./log"),
            )
            self.watermark_manager = WatermarkManager(
                enabled=True,
                mode="full",
                agent_id=agent_id,
                log_dir=watermark_log_dir,
                config={"embedding_strategy": "cyclic"}  # 循环嵌入
            )
            agent_log.info(f"Agent {agent_id}: Auto-created independent WatermarkManager with cyclic embedding")
        else:
            self.watermark_manager = watermark_manager
            
        if self.watermark_manager and WATERMARK_AVAILABLE:
            agent_log.info(f"Agent {agent_id}: Watermark enabled")
        elif watermark_manager and not WATERMARK_AVAILABLE:
            agent_log.warning(
                f"Agent {agent_id}: Watermark requested but module not available"
            )
        if user_info_template is None:
            system_message_content = self.user_info.to_system_message()
        else:
            system_message_content = self.user_info.to_custom_system_message(
                user_info_template)
        system_message = BaseMessage.make_assistant_message(
            role_name="system",
            content=system_message_content,  # system prompt
        )

        if not available_actions:
            agent_log.info("No available actions defined, using all actions.")
            self.action_tools = self.env.action.get_openai_function_list()
        else:
            all_tools = self.env.action.get_openai_function_list()
            all_possible_actions = [tool.func.__name__ for tool in all_tools]

            for action in available_actions:
                action_name = action.value if isinstance(
                    action, ActionType) else action
                if action_name not in all_possible_actions:
                    agent_log.warning(
                        f"Action {action_name} is not supported. Supported "
                        f"actions are: {', '.join(all_possible_actions)}")
            self.action_tools = [
                tool for tool in all_tools if tool.func.__name__ in [
                    a.value if isinstance(a, ActionType) else a
                    for a in available_actions
                ]
            ]
        all_tools = (tools or []) + (self.action_tools or [])
        super().__init__(
            system_message=system_message,
            model=model,
            scheduling_strategy='random_model',
            tools=all_tools,
            token_limit=16000,  # ✅ 增加到 16K 以支持更长的对话历史（适配 DeepSeek）
        )
        self.max_iteration = max_iteration
        self.interview_record = interview_record
        self.agent_graph = agent_graph
        self.test_prompt = (
            "\n"
            "Helen is a successful writer who usually writes popular western "
            "novels. Now, she has an idea for a new novel that could really "
            "make a big impact. If it works out, it could greatly "
            "improve her career. But if it fails, she will have spent "
            "a lot of time and effort for nothing.\n"
            "\n"
            "What do you think Helen should do?")

    def _build_action_context(self) -> str:
        """
        构建上下文密钥字符串（用于水印 PRG 种子）
        
        使用最近 3 个行为作为上下文，与 demo_watermark_with_deepseek.py 保持一致
        
        Returns:
            str: "like_post||create_comment||follow" 或空字符串
        """
        if not hasattr(self, '_action_history'):
            self._action_history = []
        
        window_size = 3
        recent_actions = self._action_history[-window_size:]
        return "||".join(recent_actions) if recent_actions else ""

    async def _get_action_probabilities(self, env_prompt: str) -> dict[str, float]:
        """
        通过 LLM 获取所有可用行为的概率分布
        
        策略: 让 LLM 返回 JSON 格式的概率估计
        
        Args:
            env_prompt: 环境观察描述
            
        Returns:
            dict: {"like_post": 0.3, "create_comment": 0.25, ...}
        """
        # 获取所有可用行为名称
        available_actions = [tool.func.__name__ for tool in self.action_tools]
        actions_str = ", ".join(available_actions)
        
        # 构造概率查询提示
        prompt = f"""You are observing a social media environment:
{env_prompt}

Based on this observation and your profile, estimate the probability of performing each action.
Return ONLY a JSON object with probabilities (must sum to 1.0):

Available actions: {actions_str}

Output format:
{{"action_name": probability, ...}}

Example:
{{"like_post": 0.3, "create_comment": 0.25, "follow": 0.2, "refresh": 0.25}}
"""
        
        # 调用 LLM
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=prompt
        )
        
        try:
            response = await self.astep(user_msg)
            content = response.msgs[0].content
            
            # 解析 JSON
            import json
            import re
            
            # 提取 JSON（支持多种格式）
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                probabilities = json.loads(json_match.group())
                
                # 归一化
                total = sum(probabilities.values())
                if total > 0:
                    probabilities = {k: v/total for k, v in probabilities.items()}
                
                agent_log.info(
                    f"Agent {self.social_agent_id} - Probabilities extracted: "
                    f"{probabilities}"
                )
                return probabilities
        except Exception as e:
            agent_log.warning(
                f"Agent {self.social_agent_id} - Failed to parse probabilities: {e}"
            )
        
        # 回退：均匀分布
        agent_log.warning(
            f"Agent {self.social_agent_id} - Using uniform distribution as fallback"
        )
        return {action: 1.0/len(available_actions) for action in available_actions}

    async def _execute_watermarked_action(
        self, 
        action_name: str, 
        env_prompt: str
    ) -> Any:
        """
        执行水印选定的行为
        
        策略: 让 LLM 为选定的行为生成参数并执行
        
        Args:
            action_name: 要执行的行为名称（由水印选择）
            env_prompt: 环境观察
            
        Returns:
            执行结果
        """
        # 构造强制执行的提示
        prompt = f"""You are observing a social media environment:
{env_prompt}

You MUST perform the action: {action_name}

Generate appropriate arguments for this action and execute it. Do not consider other actions.
"""
        
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=prompt
        )
        
        try:
            # 调用 LLM，它会自动选择工具并执行
            response = await self.astep(user_msg)
            
            # 记录到历史
            if not hasattr(self, '_action_history'):
                self._action_history = []
            self._action_history.append(action_name)
            
            # 验证执行结果并保存执行详情
            if 'tool_calls' in response.info:
                for tool_call in response.info['tool_calls']:
                    executed_action = tool_call.tool_name
                    args = tool_call.args
                    
                    # 🎯 保存执行详情到 _last_decision 供可视化
                    if hasattr(self, '_last_decision') and self._last_decision:
                        self._last_decision["executed_action"] = executed_action
                        self._last_decision["executed_args"] = args
                        self._last_decision["action_match"] = (executed_action == action_name)
                    
                    if executed_action == action_name:
                        agent_log.info(
                            f"Agent {self.social_agent_id} - Watermark action "
                            f"'{action_name}' executed successfully with args: {args}"
                        )
                    else:
                        agent_log.warning(
                            f"Agent {self.social_agent_id} - LLM executed "
                            f"'{executed_action}' instead of watermark-selected "
                            f"'{action_name}'"
                        )
            
            # 更新社交图谱
            if 'tool_calls' in response.info:
                for tool_call in response.info['tool_calls']:
                    self.perform_agent_graph_action(tool_call.tool_name, tool_call.args)

            return response
        except Exception as e:
            agent_log.error(
                f"Agent {self.social_agent_id} - Error executing watermarked "
                f"action '{action_name}': {e}"
            )
            raise

    async def perform_action_by_llm(self):
        """
        执行 LLM 驱动的行为决策
        
        如果启用水印:
            1. 获取行为概率分布（第1次LLM调用）
            2. 使用水印修改概率
            3. 执行选定行为（第2次LLM调用）
        否则:
            正常 LLM 流程
        """
        # Get posts:
        env_prompt = await self.env.to_text_prompt()
        
        # 🎯 水印集成点：两阶段调用法
        if self.watermark_manager and self.watermark_manager.enabled:
            try:
                agent_log.info(
                    f"Agent {self.social_agent_id} - Watermark enabled, "
                    f"using two-phase approach"
                )
                
                # === 阶段 1: 获取概率分布 ===
                agent_log.info(
                    f"Agent {self.social_agent_id} - Phase 1: Getting action probabilities"
                )
                probabilities = await self._get_action_probabilities(env_prompt)
                
                # === 阶段 2: 水印采样 ===
                agent_log.info(
                    f"Agent {self.social_agent_id} - Phase 2: Watermark sampling"
                )
                context_for_key = self._build_action_context()
                round_num = self.watermark_manager.stats.get('rounds_completed', 0)
                
                selected_action, target_list, bits_embedded, context_used = \
                    self.watermark_manager.sample_behavior_watermark(
                        probabilities=probabilities,
                        round_num=round_num,
                        context_for_key=context_for_key
                    )
                
                # 🎯 存储决策详情供可视化使用
                self._last_decision = {
                    "env_prompt": env_prompt[:500] if len(env_prompt) > 500 else env_prompt,  # 截断过长内容
                    "probabilities": probabilities,
                    "selected_action": selected_action,
                    "bits_embedded": bits_embedded,
                    "round_num": round_num,
                    "context_for_key": context_for_key
                }
                
                agent_log.info(
                    f"Agent {self.social_agent_id} - Watermark selected action: "
                    f"'{selected_action}', bits embedded: {bits_embedded}"
                )
                
                # === 阶段 3: 执行选定行为 ===
                agent_log.info(
                    f"Agent {self.social_agent_id} - Phase 3: Executing watermarked action"
                )
                response = await self._execute_watermarked_action(
                    selected_action, env_prompt
                )
                
                # 更新统计
                self.watermark_manager.stats['rounds_completed'] += 1
                
                agent_log.info(
                    f"Agent {self.social_agent_id} - Watermark integration complete"
                )
                
                return response
                
            except Exception as e:
                agent_log.error(
                    f"Agent {self.social_agent_id} - Watermark integration failed: {e}, "
                    f"falling back to normal mode"
                )
                # 回退到正常模式
                pass
        
        # 无水印模式：正常流程
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"Please perform social media actions after observing the "
                f"platform environments. Notice that don't limit your "
                f"actions for example to just like the posts. "
                f"Here is your social media environment: {env_prompt}"))
        try:
            agent_log.info(
                f"Agent {self.social_agent_id} observing environment: "
                f"{env_prompt}")
            response = await self.astep(user_msg)
            
            for tool_call in response.info['tool_calls']:
                action_name = tool_call.tool_name
                args = tool_call.args
                
                agent_log.info(f"Agent {self.social_agent_id} performed "
                               f"action: {action_name} with args: {args}")
                if action_name not in ALL_SOCIAL_ACTIONS:
                    agent_log.info(
                        f"Agent {self.social_agent_id} get the result: "
                        f"{tool_call.result}")
                # 更新社交图谱
                self.perform_agent_graph_action(action_name, args)

                return response
        except Exception as e:
            agent_log.error(f"Agent {self.social_agent_id} error: {e}")
            return e

    async def perform_test(self):
        """
        doing group polarization test for all agents.
        TODO: rewrite the function according to the ChatAgent.
        TODO: unify the test and interview function.
        """
        # user conduct test to agent
        _ = BaseMessage.make_user_message(role_name="User",
                                          content=("You are a twitter user."))
        # Test memory should not be writed to memory.
        # self.memory.write_record(MemoryRecord(user_msg,
        #                                       OpenAIBackendRole.USER))

        openai_messages, num_tokens = self.memory.get_context()

        openai_messages = ([{
            "role":
            self.system_message.role_name,
            "content":
            self.system_message.content.split("# RESPONSE FORMAT")[0],
        }] + openai_messages + [{
            "role": "user",
            "content": self.test_prompt
        }])

        agent_log.info(f"Agent {self.social_agent_id}: {openai_messages}")
        # NOTE: this is a temporary solution.
        # Camel can not stop updating the agents' memory after stop and astep
        # now.
        response = await self._aget_model_response(
            openai_messages=openai_messages, num_tokens=num_tokens)
        content = response.output_messages[0].content
        agent_log.info(
            f"Agent {self.social_agent_id} receive response: {content}")
        return {
            "user_id": self.social_agent_id,
            "prompt": openai_messages,
            "content": content
        }

    async def perform_interview(self, interview_prompt: str):
        """
        Perform an interview with the agent.
        """
        # user conduct test to agent
        user_msg = BaseMessage.make_user_message(
            role_name="User", content=("You are a twitter user."))

        if self.interview_record:
            # Test memory should not be writed to memory.
            self.update_memory(message=user_msg, role=OpenAIBackendRole.SYSTEM)

        openai_messages, num_tokens = self.memory.get_context()

        openai_messages = ([{
            "role":
            self.system_message.role_name,
            "content":
            self.system_message.content.split("# RESPONSE FORMAT")[0],
        }] + openai_messages + [{
            "role": "user",
            "content": interview_prompt
        }])

        agent_log.info(f"Agent {self.social_agent_id}: {openai_messages}")
        # NOTE: this is a temporary solution.
        # Camel can not stop updating the agents' memory after stop and astep
        # now.

        response = await self._aget_model_response(
            openai_messages=openai_messages, num_tokens=num_tokens)

        content = response.output_messages[0].content

        if self.interview_record:
            # Test memory should not be writed to memory.
            self.update_memory(message=response.output_messages[0],
                               role=OpenAIBackendRole.USER)
        agent_log.info(
            f"Agent {self.social_agent_id} receive response: {content}")

        # Record the complete interview (prompt + response) through the channel
        interview_data = {"prompt": interview_prompt, "response": content}
        result = await self.env.action.perform_action(
            interview_data, ActionType.INTERVIEW.value)

        # Return the combined result
        return {
            "user_id": self.social_agent_id,
            "prompt": openai_messages,
            "content": content,
            "success": result.get("success", False)
        }

    async def perform_action_by_hci(self) -> Any:
        print("Please choose one function to perform:")
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            agent_log.info(f"Agent {self.social_agent_id} function: "
                           f"{function_list[i].func.__name__}")

        selection = int(input("Enter your choice: "))
        if not 0 <= selection < len(function_list):
            agent_log.error(f"Agent {self.social_agent_id} invalid input.")
            return
        func = function_list[selection].func

        params = inspect.signature(func).parameters
        args = []
        for param in params.values():
            while True:
                try:
                    value = input(f"Enter value for {param.name}: ")
                    args.append(value)
                    break
                except ValueError:
                    agent_log.error("Invalid input, please enter an integer.")

        result = await func(*args)
        return result

    async def perform_action_by_data(self, func_name, *args, **kwargs) -> Any:
        func_name = func_name.value if isinstance(func_name,
                                                  ActionType) else func_name
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            if function_list[i].func.__name__ == func_name:
                func = function_list[i].func
                result = await func(*args, **kwargs)
                self.update_memory(message=BaseMessage.make_user_message(
                    role_name=OpenAIBackendRole.SYSTEM,
                    content=f"Agent {self.social_agent_id} performed "
                    f"{func_name} with args: {args} and kwargs: {kwargs}"
                    f"and the result is {result}"),
                                   role=OpenAIBackendRole.SYSTEM)
                agent_log.info(f"Agent {self.social_agent_id}: {result}")
                return result
        raise ValueError(f"Function {func_name} not found in the list.")

    def perform_agent_graph_action(
        self,
        action_name: str,
        arguments: dict[str, Any],
    ):
        r"""Remove edge if action is unfollow or add edge
        if action is follow to the agent graph.
        """
        if "unfollow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.remove_edge(self.social_agent_id, followee_id)
            agent_log.info(
                f"Agent {self.social_agent_id} unfollowed Agent {followee_id}")
        elif "follow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.add_edge(self.social_agent_id, followee_id)
            agent_log.info(
                f"Agent {self.social_agent_id} followed Agent {followee_id}")

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(agent_id={self.social_agent_id}, "
                f"model_type={self.model_type.value})")
