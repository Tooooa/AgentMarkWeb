"""
实验日志模块
职责：处理文件写入和统计计算
"""


def initialize_log_file(log_path, round_idx, total_rounds, verbose_log_path=None, enable_verbose=False):
    """
    初始化日志文件并写入轮次标题

    Args:
        log_path (str): 日志文件路径
        round_idx (int): 当前轮次索引（从 0 开始）
        total_rounds (int): 总轮次数
        verbose_log_path (str, optional): 详细日志文件路径
        enable_verbose (bool, optional): 是否启用详细日志
    """
    mode = 'w' if round_idx == 0 else 'a'
    with open(log_path, mode, encoding='utf-8') as f:
        f.write(f"\n\n{'='*30} Round {round_idx + 1}/{total_rounds} {'='*30}\n\n")

    if enable_verbose and verbose_log_path:
        verbose_mode = 'w' if round_idx == 0 else 'a'
        with open(verbose_log_path, verbose_mode, encoding='utf-8') as vf:
            vf.write(f"\n\n{'='*30} Round {round_idx + 1}/{total_rounds} - Verbose {'='*30}\n\n")


def log_round_results(log_path, round_idx, round_data):
    """
    格式化单轮数据并追加到日志

    Args:
        log_path (str): 日志文件路径
        round_idx (int): 当前轮次索引（从 1 开始）
        round_data (dict): 包含轮次数据的字典
    """
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{round_idx}:\n")
        f.write(f"  event_watermark: {round_data['event_watermark']}\n")
        f.write(f"  BEHAVIOR_TYPES: {round_data['BEHAVIOR_TYPES']}\n")
        f.write(f"  time_cost: {round_data['time_cost']}\n")
        f.write(f"  probabilities_baseline: {round_data['probabilities_baseline']}\n")
        f.write(f"  selected_behavior_baseline: {round_data['selected_behavior_baseline']}\n")
        f.write(f"  probabilities_watermark: {round_data['probabilities_watermark']}\n")
        f.write(f"  selected_behavior_watermark: {round_data['selected_behavior_watermark']}\n")
        f.write(f"  target_behavior_list: {round_data['target_behavior_list']}\n")
        f.write(f"  behaviors_match: {round_data['behaviors_match']}\n")
        f.write(f"  watermark_hit: {round_data['watermark_hit']}\n")
        f.write(f"  baseline_hit_target: {round_data['baseline_hit_target']}\n")
        
        # 如果存在差分引擎数据，也记录下来
        if 'num_bits_embedded' in round_data:
            f.write(f"  num_bits_embedded: {round_data['num_bits_embedded']}\n")
        if 'bit_index' in round_data:
            f.write(f"  bit_index: {round_data['bit_index']}\n")
        if 'context_for_key' in round_data:
            f.write(f"  context_for_key: {round_data['context_for_key']}\n")
        
        f.write("\n")


def log_long_responses(verbose_log_path, long_responses, enable_verbose=False):
    """
    记录详细的 API 响应

    Args:
        verbose_log_path (str): 详细日志文件路径
        long_responses (dict): 详细响应字典
        enable_verbose (bool, optional): 是否启用详细日志
    """
    if not enable_verbose or not verbose_log_path:
        return

    with open(verbose_log_path, 'a', encoding='utf-8') as f:
        f.write("=== Detailed Responses ===\n\n")
        for key, value in long_responses.items():
            f.write(f"{key}:\n{value}\n\n")


def log_summary(log_path, stats, round_idx, total_duration, epoch_num):
    """
    将最终统计信息写入日志

    Args:
        log_path (str): 日志文件路径
        stats (dict): 统计字典
    """
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write("=== Statistics ===\n")
        f.write(f"Round {round_idx + 1} total duration: {total_duration:.2f}s\n")
        f.write(f"Round {round_idx + 1} avg duration per epoch: {(total_duration/epoch_num):.2f}s\n\n")
        f.write(
            f"Round {round_idx + 1} different behaviors: {stats['different_behavior_count']}/{stats['total_rounds']}\n"
        )
        f.write(
            f"Round {round_idx + 1} different behavior ratio: {stats['different_behavior_ratio']:.2f}%\n"
        )
        f.write(f"Round {round_idx + 1} watermark hits: {stats['watermark_hit_count']}\n")
        f.write(f"Round {round_idx + 1} watermark hit ratio: {stats['watermark_hit_ratio']:.2f}%\n")
        f.write(f"Round {round_idx + 1} baseline hits: {stats['original_hit_count']}\n")
        f.write(f"Round {round_idx + 1} baseline hit ratio: {stats['original_hit_ratio']:.2f}%\n")

        # 如果存在差分引擎统计，记录下来
        if 'total_bits_embedded' in stats:
            f.write(f"Round {round_idx + 1} total bits embedded: {stats['total_bits_embedded']}\n")
        if 'bit_stream_usage' in stats:
            f.write(f"Round {round_idx + 1} bit stream usage: {stats['bit_stream_usage']:.2f}%\n")

        f.write("\n")


def calculate_statistics(all_rounds_data, epoch_num):
    """
    从所有轮次计算统计信息（命中率、差异）

    Args:
        all_rounds_data (list): 所有轮次数据
        epoch_num (int): 总轮次数

    Returns:
        dict: 统计字典
    """
    different_behavior_count = 0
    watermark_hit_count = 0
    original_hit_count = 0
    total_bits_embedded = 0
    
    for round_data in all_rounds_data:
        # 计算行为差异
        if not round_data["behaviors_match"]:
            different_behavior_count += 1
        
        # 计算水印命中
        if round_data["watermark_hit"]:
            watermark_hit_count += 1
        
        # 计算基线行为命中
        if round_data.get("baseline_hit_target", round_data.get("original_hit", False)):
            original_hit_count += 1
        
        # 如果存在差分引擎数据，计算嵌入的位数
        if 'num_bits_embedded' in round_data:
            total_bits_embedded += round_data['num_bits_embedded']
    
    # 计算比率
    stats = {
        'total_rounds': epoch_num,
        'different_behavior_count': different_behavior_count,
        'different_behavior_ratio': (different_behavior_count / epoch_num) * 100 if epoch_num > 0 else 0,
        'watermark_hit_count': watermark_hit_count,
        'watermark_hit_ratio': (watermark_hit_count / epoch_num) * 100 if epoch_num > 0 else 0,
        'original_hit_count': original_hit_count,
        'original_hit_ratio': (original_hit_count / epoch_num) * 100 if epoch_num > 0 else 0,
    }
    
    # 如果使用了差分引擎，添加相关统计
    if total_bits_embedded > 0:
        stats['total_bits_embedded'] = total_bits_embedded
    
    return stats


def print_statistics(stats, round_idx, total_duration, epoch_num):
    """
    将统计信息打印到控制台

    Args:
        stats (dict): 统计字典
        round_idx (int): 当前轮次索引（从 0 开始）
        total_duration (float): 总持续时间（秒）
        epoch_num (int): 每轮的轮次数
    """
    print(f"\nRound {round_idx + 1} total duration: {total_duration:.2f}s")
    print(f"Round {round_idx + 1} avg duration per epoch: {(total_duration/epoch_num):.2f}s")
    print(f"\nRound {round_idx + 1} different behaviors: {stats['different_behavior_count']}/{stats['total_rounds']}")
    print(f"Round {round_idx + 1} different behavior ratio: {stats['different_behavior_ratio']:.2f}%")
    print(f"Round {round_idx + 1} watermark hits: {stats['watermark_hit_count']}")
    print(f"Round {round_idx + 1} watermark hit ratio: {stats['watermark_hit_ratio']:.2f}%")
    print(f"Round {round_idx + 1} baseline hits: {stats['original_hit_count']}")
    print(f"Round {round_idx + 1} baseline hit ratio: {stats['original_hit_ratio']:.2f}%")

    # 如果存在差分引擎统计，打印出来
    if 'total_bits_embedded' in stats:
        print(f"Round {round_idx + 1} total bits embedded: {stats['total_bits_embedded']}")
