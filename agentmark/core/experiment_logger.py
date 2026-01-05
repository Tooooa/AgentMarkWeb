"""
Experiment logging module.
Responsibilities: handle file writes and statistics computation.
"""


def initialize_log_file(log_path, round_idx, total_rounds, verbose_log_path=None, enable_verbose=False):
    """
    Initialize log file and write round header.

    Args:
        log_path (str): Log file path
        round_idx (int): Current round index (0-based)
        total_rounds (int): Total number of rounds
        verbose_log_path (str, optional): Verbose log file path
        enable_verbose (bool, optional): Whether to enable verbose logging
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
    Format a single round's data and append to the log.

    Args:
        log_path (str): Log file path
        round_idx (int): Current round index (1-based)
        round_data (dict): Dict with round data
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
        
        # If differential-engine data exists, log it as well
        if 'num_bits_embedded' in round_data:
            f.write(f"  num_bits_embedded: {round_data['num_bits_embedded']}\n")
        if 'bit_index' in round_data:
            f.write(f"  bit_index: {round_data['bit_index']}\n")
        if 'context_for_key' in round_data:
            f.write(f"  context_for_key: {round_data['context_for_key']}\n")
        
        f.write("\n")


def log_long_responses(verbose_log_path, long_responses, enable_verbose=False):
    """
    Log detailed API responses.

    Args:
        verbose_log_path (str): Verbose log file path
        long_responses (dict): Dict of long responses
        enable_verbose (bool, optional): Whether verbose logging is enabled
    """
    if not enable_verbose or not verbose_log_path:
        return

    with open(verbose_log_path, 'a', encoding='utf-8') as f:
        f.write("=== Detailed Responses ===\n\n")
        for key, value in long_responses.items():
            f.write(f"{key}:\n{value}\n\n")


def log_summary(log_path, stats, round_idx, total_duration, epoch_num):
    """
    Write final statistics to log.

    Args:
        log_path (str): Log file path
        stats (dict): Statistics dict
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

        # If differential-engine stats exist, log them
        if 'total_bits_embedded' in stats:
            f.write(f"Round {round_idx + 1} total bits embedded: {stats['total_bits_embedded']}\n")
        if 'bit_stream_usage' in stats:
            f.write(f"Round {round_idx + 1} bit stream usage: {stats['bit_stream_usage']:.2f}%\n")

        f.write("\n")


def calculate_statistics(all_rounds_data, epoch_num):
    """
    Compute statistics (hit rates, differences) from all rounds.

    Args:
        all_rounds_data (list): All round data
        epoch_num (int): Total epochs

    Returns:
        dict: Statistics dict
    """
    different_behavior_count = 0
    watermark_hit_count = 0
    original_hit_count = 0
    total_bits_embedded = 0
    
    for round_data in all_rounds_data:
        # Count behavior differences
        if not round_data["behaviors_match"]:
            different_behavior_count += 1
        
        # Count watermark hits
        if round_data["watermark_hit"]:
            watermark_hit_count += 1
        
        # Count baseline behavior hits
        if round_data.get("baseline_hit_target", round_data.get("original_hit", False)):
            original_hit_count += 1
        
        # If differential-engine data exists, count embedded bits
        if 'num_bits_embedded' in round_data:
            total_bits_embedded += round_data['num_bits_embedded']
    
    # Compute ratios
    stats = {
        'total_rounds': epoch_num,
        'different_behavior_count': different_behavior_count,
        'different_behavior_ratio': (different_behavior_count / epoch_num) * 100 if epoch_num > 0 else 0,
        'watermark_hit_count': watermark_hit_count,
        'watermark_hit_ratio': (watermark_hit_count / epoch_num) * 100 if epoch_num > 0 else 0,
        'original_hit_count': original_hit_count,
        'original_hit_ratio': (original_hit_count / epoch_num) * 100 if epoch_num > 0 else 0,
    }
    
    # If differential engine used, add related stats
    if total_bits_embedded > 0:
        stats['total_bits_embedded'] = total_bits_embedded
    
    return stats


def print_statistics(stats, round_idx, total_duration, epoch_num):
    """
    Print statistics to console.

    Args:
        stats (dict): Statistics dict
        round_idx (int): Current round index (0-based)
        total_duration (float): Total duration (seconds)
        epoch_num (int): Epochs per round
    """
    print(f"\nRound {round_idx + 1} total duration: {total_duration:.2f}s")
    print(f"Round {round_idx + 1} avg duration per epoch: {(total_duration/epoch_num):.2f}s")
    print(f"\nRound {round_idx + 1} different behaviors: {stats['different_behavior_count']}/{stats['total_rounds']}")
    print(f"Round {round_idx + 1} different behavior ratio: {stats['different_behavior_ratio']:.2f}%")
    print(f"Round {round_idx + 1} watermark hits: {stats['watermark_hit_count']}")
    print(f"Round {round_idx + 1} watermark hit ratio: {stats['watermark_hit_ratio']:.2f}%")
    print(f"Round {round_idx + 1} baseline hits: {stats['original_hit_count']}")
    print(f"Round {round_idx + 1} baseline hit ratio: {stats['original_hit_ratio']:.2f}%")

    # If differential-engine stats exist, print them
    if 'total_bits_embedded' in stats:
        print(f"Round {round_idx + 1} total bits embedded: {stats['total_bits_embedded']}")
