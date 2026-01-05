"""
Log parser module.
Responsibilities: parse experiment logs and extract data for embedding/decoding.
"""

import re
import yaml


def parse_log_files(log_path, verbose_log_path=None):
    """
    Parse summary and verbose logs and merge their data.

    Args:
        log_path (str): Summary log file path (watermark_log.txt)
        verbose_log_path (str, optional): Verbose log file path (watermark_verbose.log)

    Returns:
        list: List of dicts, each representing one round.
        
    Example:
        >>> rounds = parse_log_files('log/watermark_log.txt', 'log/watermark_verbose.log')
        >>> print(f"Parsed {len(rounds)} rounds")
        >>> print(f"Round 1 selected watermark behavior: {rounds[0]['selected_behavior_watermark']}")
    """
    # --- 1. Parse summary log ---
    print(f"Parsing summary log: {log_path}")
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: log file not found {log_path}")
        return []

    rounds_data = {}
    
    # Find each round block with regex
    # Format: "number:\n  key: value\n  key: value..."
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
                
                # Use PyYAML to safely parse dict/list strings
                try:
                    parsed_value = yaml.safe_load(value)
                    data[key.strip()] = parsed_value
                except yaml.YAMLError:
                    # If YAML parsing fails, keep raw string
                    data[key.strip()] = value
                    
        rounds_data[round_num] = data
    
    print(f"Parsed {len(rounds_data)} rounds from summary log")
        
    # --- 2. Parse verbose log for behavior_response_watermark ---
    if verbose_log_path:
        print(f"Parsing verbose log: {verbose_log_path}")
        
        try:
            with open(verbose_log_path, 'r', encoding='utf-8') as f:
                verbose_content = f.read()
        except FileNotFoundError:
            print(f"Warning: verbose log not found {verbose_log_path}, skipping behavior descriptions")
            verbose_content = ""

        if verbose_content:
            # Regex match for round_X_behavior_response_watermark: and its content
            # Format: "round_<num>_behavior_response_watermark:\n> behavior type:..."
            response_blocks = re.findall(
                r'round_(\d+)_behavior_response_watermark:\n((?:>.*\n?)+)', 
                verbose_content
            )
            
            for round_num_str, response_text in response_blocks:
                round_num = int(round_num_str)
                if round_num in rounds_data:
                    # Merge multi-line '>' descriptions into a single string
                    cleaned_response = re.sub(r'>\s*', '', response_text).strip()
                    rounds_data[round_num]['behavior_response_watermark'] = cleaned_response
            
            print(f"Extracted {len(response_blocks)} behavior descriptions from verbose log")
    
    # --- 3. Convert to ordered list ---
    sorted_rounds_list = [rounds_data[i] for i in sorted(rounds_data.keys())]
    
    print(f"Log parsing complete, total rounds: {len(sorted_rounds_list)}")
    
    return sorted_rounds_list


def validate_round_data(round_data, round_num):
    """
    Validate completeness of a single round.
    
    Args:
        round_data (dict): Round data dict
        round_num (int): Round index (for error reporting)
        
    Returns:
        tuple: (is_valid, missing_keys)
        
    Example:
        >>> valid, missing = validate_round_data(rounds[0], 1)
        >>> if not valid:
        ...     print(f"Round 1 missing fields: {missing}")
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
        print(f"Warning: round {round_num} incomplete, missing fields: {missing_keys}")
    
    return is_valid, missing_keys


def extract_statistics_from_log(log_path):
    """
    Extract statistics from summary log.
    
    Args:
        log_path (str): Summary log file path
        
    Returns:
        dict: Stats dict with duration, hit rates, embedded bits, etc.
        
    Example:
        >>> stats = extract_statistics_from_log('log/watermark_log.txt')
        >>> print(f"Watermark hit rate: {stats['watermark_hit_rate']}%")
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: log file not found {log_path}")
        return {}
    
    stats = {}
    
    # Extract statistics
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
            # Try to convert to number
            try:
                if '.' in value:
                    stats[key] = float(value)
                else:
                    stats[key] = int(value)
            except ValueError:
                stats[key] = value
    
    return stats
