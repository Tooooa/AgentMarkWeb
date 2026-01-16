"""
模型交互模块（Agent Simulator）
职责：封装所有 LLM API 交互逻辑
"""

from .prompt_utils import format_behaviors_list, generate_behaviors_example
from .parser_utils import extract_probabilities


def get_behavior_probabilities(client, model, role_config, event, behaviors, probability_template):
    """
    根据事件获取行为的概率分布

    Args:
        client: OpenAI 客户端实例
        model: 模型名称
        role_config: 角色配置（名称、简介、系统提示）
        event: 格式化的事件文本
        behaviors: 行为类型列表，例如 ['like', 'favorite', 'share', ...]
        probability_template: 概率估计的提示模板

    Returns:
        tuple: (probabilities_dict, raw_response_text)
            - probabilities_dict: 概率字典，例如 {'like': 0.3, 'favorite': 0.2, ...}
            - raw_response_text: 原始 API 响应文本
    """
    name = role_config['name']
    profile = role_config['profile']
    
    # 构建概率提示
    probability_prompt = probability_template.format(
        name=name,
        event=event,
        behaviors=format_behaviors_list(behaviors),
        behaviors_example=generate_behaviors_example(behaviors)
    )
    
    print("概率提示：")
    print(probability_prompt)
    
    # 调用 API 获取行为概率
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": role_config["system_prompt"].format(name=name, profile=profile)
            },
            {
                "role": "user", 
                "content": probability_prompt
            }
        ]
    )
    
    # 获取响应文本
    response_text = response.choices[0].message.content
    print("\n概率响应：")
    print(response_text)
    
    # 提取概率
    probabilities = extract_probabilities(response_text, behaviors)
    
    if probabilities:
        print("\n提取的概率：")
        print(probabilities)
    else:
        print("\n警告：提取概率失败")
    
    return probabilities, response_text


def get_behavior_description(client, model, role_config, event, behavior, behavior_template):
    """
    根据事件获取所选行为的详细描述

    Args:
        client: OpenAI 客户端实例
        model: 模型名称
        role_config: 角色配置
        event: 格式化的事件文本
        behavior: 选定的行为，例如 "like"
        behavior_template: 行为描述的提示模板

    Returns:
        str: 模型生成的行为描述
    """
    name = role_config['name']
    profile = role_config['profile']
    
    # 构建行为描述提示
    behavior_prompt = behavior_template.format(
        name=name,
        event=event,
        behavior=behavior
    )
    
    print(f"\n行为提示（行为：{behavior}）：")
    print(behavior_prompt)
    
    # 调用 API 获取行为描述
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": role_config["system_prompt"].format(name=name, profile=profile)
            },
            {
                "role": "user", 
                "content": behavior_prompt
            }
        ]
    )
    
    behavior_description = response.choices[0].message.content
    print("\n行为描述：")
    print(behavior_description)
    
    return behavior_description
