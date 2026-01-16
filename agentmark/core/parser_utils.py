"""
解析工具模块
职责：从 LLM 响应中解析结构化数据
"""

import re
import json


def extract_probabilities(response_text, behaviors):
    """
    从 API 响应文本中提取行为概率

    支持的策略：
    1. 代码块：解析 ```json``` 块（首选）
    2. JSON 解析：解析响应中的第一个 JSON 对象
    3. 正则表达式回退：使用正则表达式匹配每个行为概率

    扩展支持 ALFWorld 命令格式：
    - 带特殊字符的命令（例如 "go to cabinet 1", "take soapbar 1 from countertop 1"）
    - 部分概率提取（返回部分字典）
    - 包含简要分析部分的响应

    Args:
        response_text (str): API 响应文本
        behaviors (list): 行为列表

    Returns:
        dict: 概率字典，如果提取失败则返回 None
              如果只找到部分概率，则返回部分字典

    Example:
        >>> response = 'Brief Analysis: xxx\n```json\n{"like": 0.3}\n```'
        >>> behaviors = ["like", "favorite", "share"]
        >>> extract_probabilities(response, behaviors)
        {'like': 0.3}

        >>> response = 'Based on the scene, probabilities are {"like": 0.3, "favorite": 0.2, "share": 0.5}'
        >>> behaviors = ["like", "favorite", "share"]
        >>> extract_probabilities(response, behaviors)
        {'like': 0.3, 'favorite': 0.2, 'share': 0.5}
    """
    # 策略 0：优先使用 JSON 代码块
    try:
        code_block_pattern = r"```json\s*\n([\s\S]*?)\n```"
        code_match = re.search(code_block_pattern, response_text)
        if code_match:
            json_text = code_match.group(1).strip()
            parsed = json.loads(json_text)
            
            # 检查是否所有行为键都存在
            if all(b in parsed for b in behaviors):
                return {b: float(parsed[b]) for b in behaviors}
            
            # 支持部分提取
            partial_result = {}
            for b in behaviors:
                if b in parsed:
                    try:
                        partial_result[b] = float(parsed[b])
                    except (ValueError, TypeError):
                        continue
            
            if partial_result:
                return partial_result
    except Exception:
        # 解析失败，尝试下一个策略
        pass
    
    # 策略 1：解析第一个 JSON 对象（更健壮）
    try:
        # 查找第一个大括号包裹的 JSON 块
        m = re.search(r"\{[\s\S]*?\}", response_text)
        if m:
            json_text = m.group(0)
            # 模型可能使用单引号；尝试转换为有效 JSON
            json_text_fixed = json_text.replace("'", '"')
            parsed = json.loads(json_text_fixed)
            
            # 检查是否所有行为键都存在
            if all(b in parsed for b in behaviors):
                return {b: float(parsed[b]) for b in behaviors}
            
            # 支持部分提取（ALFWorld 特定）
            partial_result = {}
            for b in behaviors:
                if b in parsed:
                    try:
                        partial_result[b] = float(parsed[b])
                    except (ValueError, TypeError):
                        continue
            
            if partial_result:
                return partial_result
    except Exception:
        # 解析失败，尝试正则表达式回退
        pass

    # 策略 2：每个行为使用正则表达式（支持部分匹配）
    partial_result = {}
    for behavior in behaviors:
        # 为 ALFWorld 命令转义特殊字符
        escaped_behavior = re.escape(behavior)
        # 模式："behavior": 0.5 或 'behavior': 0.5
        pattern = r'["\']' + escaped_behavior + r'["\']\s*:\s*([0-9]*\.?[0-9]+)'
        match = re.search(pattern, response_text)
        if match:
            try:
                partial_result[behavior] = float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    if partial_result:
        return partial_result

    # 策略 3：匹配完整的行为集（传统回退）
    pattern_parts = []
    for behavior in behaviors:
        pattern_parts.append(r'"' + re.escape(behavior) + r'"\s*:\s*([0-9]*\.?[0-9]+)')

    probability_pattern = r'\{[\s\S]*' + r'.*'.join(pattern_parts) + r'[\s\S]*\}'
    match = re.search(probability_pattern, response_text)
    if match:
        return {
            behavior: float(match.group(i + 1))
            for i, behavior in enumerate(behaviors)
        }

    return None



def extract_and_normalize_probabilities(response_text, admissible_commands, logger=None):
    """
    从 LLM 响应中提取并归一化概率（多级回退）

    三级回退策略：
    1. 策略 1：提取所有概率并归一化
    2. 策略 2：部分归一化，缺失的设为 0
    3. 策略 3：完全失败时使用均匀分布

    所有回退使用都会被记录

    Args:
        response_text (str): LLM 响应文本
        admissible_commands (list): 可接受的命令
        logger (logging.Logger, optional): 用于记录回退使用的日志记录器

    Returns:
        dict: 归一化的概率字典 {command: probability}

    Example:
        >>> response = '{"go to cabinet 1": 0.3, "take soapbar 1": 0.7}'
        >>> commands = ["go to cabinet 1", "take soapbar 1", "open cabinet 1"]
        >>> extract_and_normalize_probabilities(response, commands)
        {'go to cabinet 1': 0.3, 'take soapbar 1': 0.7, 'open cabinet 1': 0.0}
    """
    # 尝试提取概率
    extracted = extract_probabilities(response_text, admissible_commands)
    
    # 策略 1：完全提取并归一化
    if extracted and len(extracted) == len(admissible_commands):
        total = sum(extracted.values())
        if total > 0:
            normalized = {k: v / total for k, v in extracted.items()}
            if logger:
                logger.debug(f"提取并归一化了所有 {len(extracted)} 个概率")
            return normalized
        else:
            # 所有概率都为 0；使用均匀分布
            if logger:
                logger.warning("提取的概率总和为 0；使用均匀分布回退")
            uniform_prob = 1.0 / len(admissible_commands)
            return {cmd: uniform_prob for cmd in admissible_commands}
    
    # 策略 2：部分归一化，缺失的设为 0
    elif extracted and len(extracted) > 0:
        if logger:
            logger.warning(
                f"仅提取了 {len(extracted)}/{len(admissible_commands)} 个概率；"
                "使用部分归一化"
            )
        
        # 检查不在可接受列表中的动作（已过滤）
        filtered_actions = {k: v for k, v in extracted.items() if k not in admissible_commands}
        if filtered_actions and logger:
            logger.warning(f"LLM 动作不在可接受列表中（已过滤）: {filtered_actions}")
        
        # 仅保留可接受的命令
        valid_extracted = {k: v for k, v in extracted.items() if k in admissible_commands}
        
        # 归一化有效提取的概率
        total = sum(valid_extracted.values())
        if total > 0:
            normalized = {k: v / total for k, v in valid_extracted.items()}
            if logger and filtered_actions:
                # 显示归一化前后
                logger.info(f"归一化前（LLM 原始）: {extracted}")
                logger.info(f"归一化后（过滤+重新分配）: {normalized}")
        else:
            # 所有提取的概率都为 0；分配均匀概率
            uniform_prob = 1.0 / len(valid_extracted)
            normalized = {k: uniform_prob for k in valid_extracted.keys()}
        
        # 将缺失的命令设为 0
        for cmd in admissible_commands:
            if cmd not in normalized:
                normalized[cmd] = 0.0
        
        if logger:
            logger.info(f"提取的有效动作: {list(valid_extracted.keys())}")
            logger.info(f"缺失的动作（设为 0）: {[c for c in admissible_commands if c not in extracted]}")
        
        return normalized
    
    # 策略 3：完全失败时使用均匀分布
    else:
        if logger:
            logger.error(
                f"未能提取任何概率；使用均匀分布回退。"
                f"响应文本: {response_text[:200]}..."
            )
        
        uniform_prob = 1.0 / len(admissible_commands)
        return {cmd: uniform_prob for cmd in admissible_commands}
