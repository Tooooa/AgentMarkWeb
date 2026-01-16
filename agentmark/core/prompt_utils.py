"""
提示工具模块
目的：为 LLM 提示生成提供格式化辅助函数
"""


def format_behaviors_list(behaviors):
    """
    将行为列表格式化为项目符号列表字符串

    Args:
        behaviors (list): 行为列表

    Returns:
        str: 格式化的行为列表字符串

    Example:
        >>> behaviors = ["like", "save", "share"]
        >>> format_behaviors_list(behaviors)
        '- like\\n- save\\n- share'
    """
    return "\n".join([f"- {behavior}" for behavior in behaviors])


def generate_behaviors_example(behaviors):
    """
    为行为概率生成示例 JSON 字符串

    Args:
        behaviors (list): 行为列表

    Returns:
        str: 示例 JSON 字符串

    Example:
        >>> behaviors = ["like", "save", "share"]
        >>> generate_behaviors_example(behaviors)
        '{ "like": <probability of like>, "save": <probability of save>, "share": <probability of share> }'
    """
    example_dict = {behavior: f"<probability of {behavior}>" for behavior in behaviors}
    return "{ " + ", ".join([f'"{k}": {v}' for k, v in example_dict.items()]) + " }"
