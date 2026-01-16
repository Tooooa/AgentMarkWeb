"""
日志解析器模块
职责：解析实验日志并提取数据用于嵌入/解码
"""

import re
import yaml


def parse_log_files(log_path, verbose_log_path=None):
    """
    解析摘要和详细日志并合并它们的数据

    Args:
        log_path (str): 摘要日志文件路径（watermark_log.txt）
        verbose_log_path (str, optional): 详细日志文件路径（watermark_verbose.log）

    Returns:
        list: 字典列表，每个代表一轮
        
    Example:
        >>> rounds = parse_log_files('log/watermark_log.txt', 'log/watermark_verbose.log')
        >>> print(f"解析了 {len(rounds)} 轮")
        >>> print(f"第 1 轮选定的水印行为：{rounds[0]['selected_behavior_watermark']}")
    """
    # --- 1. 解析摘要日志 ---
    print(f"解析摘要日志：{log_path}")
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"错误：未找到日志文件 {log_path}")
        return []

    rounds_data = {}
    
    # 使用正则表达式查找每个轮次块
    # 格式："number:\n  key: value\n  key: value..."
    round_blocks = re.findall(r'^(\d+):\n((?:  .*\n)+)', log_content, re.MULTILINE)

    for round_num_str, block_content in round_blocks:
        round_num = int(round_num_str)
        data = {}
        lines = block_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                value = value.strip()
                
                # 使用 PyYAML 安全解析字典/列表字符串
                try:
                    parsed_value = yaml.safe_load(value)
                    data[key.strip()] = parsed_value
                except yaml.YAMLError:
                    # 如果 YAML 解析失败，保留原始字符串
                    data[key.strip()] = value
                    
        rounds_data[round_num] = data
    
    print(f"从摘要日志解析了 {len(rounds_data)} 轮")
        
    # --- 2. 解析详细日志以获取 behavior_response_watermark ---
    if verbose_log_path:
        print(f"解析详细日志：{verbose_log_path}")
        
        try:
            with open(verbose_log_path, 'r', encoding='utf-8') as f:
                verbose_content = f.read()
        except FileNotFoundError:
            print(f"警告：未找到详细日志 {verbose_log_path}，跳过行为描述")
            verbose_content = ""

        if verbose_content:
            # 正则匹配 round_X_behavior_response_watermark: 及其内容
            # 格式："round_<num>_behavior_response_watermark:\n> behavior type:..."
            response_blocks = re.findall(
                r'round_(\d+)_behavior_response_watermark:\n((?:>.*\n?)+)', 
                verbose_content
            )
            
            for round_num_str, response_text in response_blocks:
                round_num = int(round_num_str)
                if round_num in rounds_data:
                    # 将多行 '>' 描述合并为单个字符串
                    cleaned_response = re.sub(r'>\s*', '', response_text).strip()
                    rounds_data[round_num]['behavior_response_watermark'] = cleaned_response
            
            print(f"从详细日志提取了 {len(response_blocks)} 个行为描述")
    
    # --- 3. 转换为有序列表 ---
    sorted_rounds_list = [rounds_data[i] for i in sorted(rounds_data.keys())]
    
    print(f"日志解析完成，总轮次：{len(sorted_rounds_list)}")
    
    return sorted_rounds_list


def validate_round_data(round_data, round_num):
    """
    验证单轮的完整性
    
    Args:
        round_data (dict): 轮次数据字典
        round_num (int): 轮次索引（用于错误报告）
        
    Returns:
        tuple: (is_valid, missing_keys)
        
    Example:
        >>> valid, missing = validate_round_data(rounds[0], 1)
        >>> if not valid:
        ...     print(f"第 1 轮缺少字段：{missing}")
    """
    required_keys = [
        'probabilities_watermark',
        'selected_behavior_watermark',
        'behavior_response_watermark'
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in round_data or round_data[key] is None:
            missing_keys.append(key)
    
    is_valid = len(missing_keys) == 0
    
    if not is_valid:
        print(f"警告：第 {round_num} 轮不完整，缺少字段：{missing_keys}")
    
    return is_valid, missing_keys


def extract_statistics_from_log(log_path):
    """
    从摘要日志中提取统计信息
    
    Args:
        log_path (str): 摘要日志文件路径
        
    Returns:
        dict: 包含持续时间、命中率、嵌入位数等的统计字典
        
    Example:
        >>> stats = extract_statistics_from_log('log/watermark_log.txt')
        >>> print(f"水印命中率：{stats['watermark_hit_rate']}%")
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"错误：未找到日志文件 {log_path}")
        return {}
    
    stats = {}
    
    # 提取统计信息
    patterns = {
        'total_time': r'Round \d+ total duration: ([\\d.]+)s',
        'avg_time': r'Round \d+ avg duration per epoch: ([\\d.]+)s',
        'behavior_diff_rate': r'Round \d+ different behavior ratio: ([\\d.]+)%',
        'watermark_hit_rate': r'Round \d+ watermark hit ratio: ([\\d.]+)%',
        'original_hit_rate': r'Round \d+ baseline hit ratio: ([\\d.]+)%',
        'total_bits_embedded': r'Round \d+ total bits embedded: (\\d+)',
        'bit_usage_rate': r'Round \d+ bit stream usage: ([\\d.]+)%',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, log_content)
        if match:
            value = match.group(1)
            # 尝试转换为数字
            try:
                if '.' in value:
                    stats[key] = float(value)
                else:
                    stats[key] = int(value)
            except ValueError:
                stats[key] = value
    
    return stats
