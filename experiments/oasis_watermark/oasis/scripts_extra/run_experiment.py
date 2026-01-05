"""
Main Watermark Experiment File
Responsibilities: Experiment flow control - schedules modules, keeping code concise and clear
"""

import json
import time
import random
from pathlib import Path
from openai import OpenAI
import sys
import os
# Add global project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3] # experiments/oasis_watermark/oasis/scripts_extra -> repository root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# REPO_ROOT is already defined as PROJECT_ROOT above
REPO_ROOT = PROJECT_ROOT
DATA_DIR = REPO_ROOT / 'agentmark' / 'data'
OUTPUT_DIR = REPO_ROOT / 'output' / 'social_media_experiment'
LOG_DIR = OUTPUT_DIR / 'log'

# Import event generator
from experiments.oasis_watermark.oasis.scripts.analysis.event_generate import generate_video_event, format_video_event_to_text
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Import functional modules
from agentmark.core.agent_simulator import get_behavior_probabilities, get_behavior_description
from agentmark.core.watermark_sampler import sample_behavior, sample_behavior_differential
from agentmark.core.experiment_logger import (
    initialize_log_file,
    log_round_results,
    log_long_responses,
    log_summary,
    calculate_statistics,
    print_statistics
)
from agentmark.core.coding_utils import encode_payload


def extract_operation_description(behavior_response):
    """
    Extract the operation description part from the behavior response.
    
    Args:
        behavior_response (str): The complete behavior description.
        
    Returns:
        str: Extracted operation description, or truncated original text (to 200 chars) if extraction fails.
        
    Example:
        Input:
        > Behavior Type: Like
        > Operation Description: Quickly clicked the like button
        > Behavior Object: Video
        > Additional Info: ...
        
        Output:
        Quickly clicked the like button
    """
    import re
    
    # Try to match "Operation Description: xxx" pattern (handling both straight and Chinese colon)
    match = re.search(r'>\s*Operation Description[:：]\s*(.+?)(?:\n|$)', behavior_response, re.IGNORECASE)
    # Fallback to Chinese pattern if English match fails, though we expect inputs to be consistent
    if not match:
        match = re.search(r'>\s*操作描述[:：]\s*(.+?)(?:\n|$)', behavior_response)

    if match:
        operation_desc = match.group(1).strip()
        return operation_desc
    
    # If extraction fails, return truncated original text
    return behavior_response[:200] if len(behavior_response) > 200 else behavior_response


def load_bit_stream(file_path=None):
    """Load secret information bit stream"""
    if file_path is None:
        file_path = DATA_DIR / 'bit_stream.txt'
    else:
        file_path = Path(file_path)
    
    try:
        with open(file_path, 'r') as f:
            bit_stream = f.read().strip()
        print(f"Successfully loaded bit stream, length: {len(bit_stream)} bits")
        print(f"File path: {file_path}")
        return bit_stream
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
        # Generate temporary random bit stream
        bit_stream = ''.join(random.choice('01') for _ in range(4096))
        print("Generated temporary random bit stream (4096 bits)")
        return bit_stream


def run_single_experiment_round(
    client, 
    config, 
    role_config, 
    model,
    probability_template,
    behavior_template,
    BEHAVIOR_TYPES,
    seed,
    bit_stream,
    epoch_num,
    message_to_embed=None,
    watermark_config=None
):
    """
    Execute a single full experiment round
    
    Returns:
        tuple: (all_responses, statistics)
    """
    print(f"\n{'='*50}")
    print("Starting new experiment round")
    print(f"{'='*50}\n")
    
    # Initialize history records
    history_events = []
    history_responses = []
    watermark_history_events = []
    watermark_history_responses = []
    
    # Initialize bit stream index
    bit_index = 0
    
    # Store data for all rounds
    all_responses = {
        "rounds": [],
        "long_responses": {}
    }
    
    # Inner loop: Generate multiple video events
    for i in range(epoch_num):
        loop_start_time = time.time()
        
        print(f"\n{'─'*50}")
        print(f"Video {i+1}/{epoch_num}")
        print(f"{'─'*50}\n")
        
        # ============ Step 1: Data Generation ============
        video_event_watermark = generate_video_event(
            client=client,
            config=config,
            history_events=watermark_history_events,
            history_responses=watermark_history_responses
        )
        event_watermark = format_video_event_to_text(video_event_watermark)
        watermark_history_events.append(event_watermark)
        
        print("Video Content:")
        print(event_watermark)
        
        # ============ Step 2: Model Interaction - Get Probabilities ============
        # Get baseline probabilities
        probabilities, probability_response = get_behavior_probabilities(
            client=client,
            model=model,
            role_config=role_config,
            event=event_watermark,
            behaviors=BEHAVIOR_TYPES,
            probability_template=probability_template
        )
        
        # Get watermark probabilities (using same probabilities here, but could be different)
        probabilities_watermark, probability_response_watermark = get_behavior_probabilities(
            client=client,
            model=model,
            role_config=role_config,
            event=event_watermark,
            behaviors=BEHAVIOR_TYPES,
            probability_template=probability_template
        )
        
        if not probabilities or not probabilities_watermark:
            print("Failed to extract probability data, skipping this round")
            continue
        
        # ============ Step 3: Watermark Sampling - Baseline Version ============
        selected_behavior = sample_behavior(
            probabilities=probabilities, 
            seed=seed, 
            round_num=i
        )
        print(f"\nBaseline selected behavior: {selected_behavior}")
        
        # Get baseline behavior description
        behavior_response = get_behavior_description(
            client=client,
            model=model,
            role_config=role_config,
            event=event_watermark,
            behavior=selected_behavior,
            behavior_template=behavior_template
        )
        history_responses.append(behavior_response)
        
        # ============ Step 4: Watermark Sampling - Differential Engine Version ============
        print(f"\nStarting watermark embedding (Current bit index: {bit_index}/{len(bit_stream)})")
        
        if bit_index >= len(bit_stream):
            print("Warning: Bit stream exhausted, no further information will be embedded")
        
        # Build explicit context string (sliding window: last 3 responses, keep only operation description)
        window_size = 3
        recent_responses = watermark_history_responses[-window_size:] if len(watermark_history_responses) > 0 else []
        
        # Extract operation description from each response
        operation_descriptions = [extract_operation_description(resp) for resp in recent_responses]
        context_for_key = "||".join(operation_descriptions) if operation_descriptions else ""
        
        print(f"Context key basis (Last {window_size} operation descriptions): {len(operation_descriptions)} items")
        if operation_descriptions:
            print(f"   Latest operation: {operation_descriptions[-1][:50]}...")
        
        # Select bit stream based on embedding strategy
        # cyclic: cyclically embed the encoded message
        # once: sequential read of original bit stream (embed only once)
        if watermark_config.get('embedding_strategy') == 'cyclic' and message_to_embed:
            # Cyclic embedding: create sufficiently long cyclic bit stream
            # Estimate required length (Conservative estimate: max embed log2(6) approx 3 bits per round)
            max_possible_bits = epoch_num * 3
            num_repeats = (max_possible_bits // len(message_to_embed)) + 2
            effective_bit_stream = message_to_embed * num_repeats
            effective_bit_index = bit_index
        else:
            # Sequential embedding: use original bit stream
            effective_bit_stream = bit_stream
            effective_bit_index = bit_index
        
        selected_behavior_watermark, target_behavior_list, num_bits, context_used = sample_behavior_differential(
            probabilities=probabilities_watermark,
            bit_stream=effective_bit_stream,
            bit_index=effective_bit_index,
            context_for_key=context_for_key,  # <--- Pass explicit context string
            seed=seed,
            round_num=i
        )
        bit_index += num_bits
        
        print(f"\nWatermark selected behavior: {selected_behavior_watermark}")
        print(f"Target behavior list: {target_behavior_list}")
        print(f"Embedded {num_bits} bits this round, current index: {bit_index}/{len(bit_stream)}")
        
        # Get watermark behavior description
        behavior_response_watermark = get_behavior_description(
            client=client,
            model=model,
            role_config=role_config,
            event=event_watermark,
            behavior=selected_behavior_watermark,
            behavior_template=behavior_template
        )
        # Immediately add to history after getting description
        # This ensures next round key generation uses the latest context
        watermark_history_responses.append(behavior_response_watermark)
        
        # ============ Step 5: Log Data ============
        loop_end_time = time.time()
        loop_duration = loop_end_time - loop_start_time
        
        round_data = {
            "event_watermark": event_watermark,
            "BEHAVIOR_TYPES": BEHAVIOR_TYPES,
            "time_cost": f"{loop_duration:.2f}",
            "probabilities_baseline": probabilities,
            "selected_behavior_baseline": selected_behavior,
            "probabilities_watermark": probabilities_watermark,
            "selected_behavior_watermark": selected_behavior_watermark,
            "target_behavior_list": target_behavior_list,
            "behaviors_match": selected_behavior == selected_behavior_watermark,
            "watermark_hit": selected_behavior_watermark in target_behavior_list,
            "baseline_hit_target": selected_behavior in target_behavior_list,
            "num_bits_embedded": num_bits,
            "bit_index": bit_index,
            "context_for_key": context_used
        }
        
        all_responses["rounds"].append(round_data)
        
        # Log long responses
        all_responses["long_responses"][f"round_{i+1}_behavior_response_baseline"] = behavior_response
        all_responses["long_responses"][f"round_{i+1}_probability_response_baseline"] = probability_response
        all_responses["long_responses"][f"round_{i+1}_behavior_response_watermark"] = behavior_response_watermark
        all_responses["long_responses"][f"round_{i+1}_probability_response_watermark"] = probability_response_watermark
        
        print(f"\nLoop duration: {loop_duration:.2f}s")
        time.sleep(1)  # Avoid hitting rate limits
    
    # Calculate statistics
    statistics = calculate_statistics(all_responses["rounds"], epoch_num)
    statistics['bit_stream_length'] = len(bit_stream)
    statistics['final_bit_index'] = bit_index
    statistics['bit_stream_usage'] = (bit_index / len(bit_stream)) * 100 if len(bit_stream) > 0 else 0
    
    return all_responses, statistics


def main():
    """Main function"""
    
    print("="*60)
    print("AgentMark Watermark Experiment")
    print("="*60)
    
    # ============ Load Config ============
    config_path = PROJECT_ROOT / 'experiments' / 'oasis_watermark' / 'oasis' / 'config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Expand environment variables
    for k, v in config.items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            var_name = v[2:-1]
            env_val = os.environ.get(var_name)
            if env_val:
                config[k] = env_val
            else:
                config[k] = "" # Fallback
                
    print(f"Config file: {config_path}")
    
    # Initialize OpenAI Client
    client = OpenAI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )
    
    # Load config parameters
    BEHAVIOR_TYPES = ['Like', 'Favorite', 'Repost', 'Comment', 'View', 'Download']
    role_config = config['role_config']
    model = config['model']
    probability_template = config['probability_template']
    behavior_template = config['behavior_template']
    seed = config['seed']
    enable_verbose_log = config.get('enable_verbose_log', False)
    
    # Load watermark config
    watermark_config = config.get('watermark_config', {
        "payload_bit_length": 8,
        "ecc_method": "none",
        "embedding_strategy": "once"
    })
    
    # Log paths - convert to new directory structure
    standard_log_path = LOG_DIR / 'watermark_log.txt'
    verbose_log_path = LOG_DIR / 'watermark_verbose.log' if enable_verbose_log else None
    
    # Read experiment parameters from config
    epoch_num = config.get('watermark_epoch_num', 50)
    total_rounds = config.get('experiment_rounds', 10)
    
    print(f"\nExperiment Config:")
    print(f"   - Model: {model}")
    print(f"   - Role: {role_config['name']}")
    print(f"\nWatermark Config:")
    print(f"   - Payload Length: {watermark_config.get('payload_bit_length')} bits")
    print(f"   - ECC Method: {watermark_config.get('ecc_method')}")
    print(f"   - Embedding Strategy: {watermark_config.get('embedding_strategy')}")
    print(f"   - Videos per Round: {epoch_num}")
    print(f"   - Total Rounds: {total_rounds}")
    print(f"   - Behavior Types: {BEHAVIOR_TYPES}")
    
    # ============ Load Bit Stream ============
    bit_stream = load_bit_stream()
    
    # ============ Prepare Encoded Message ============
    payload_bit_length = watermark_config.get('payload_bit_length', 8)
    payload_bits = bit_stream[:payload_bit_length].strip()
    
    if len(payload_bits) < payload_bit_length:
        print(f"Warning: Bit stream insufficient length {payload_bit_length} bits, actual {len(payload_bits)} bits")
        # Pad to required length
        payload_bits = payload_bits.ljust(payload_bit_length, '0')
    
    try:
        message_to_embed = encode_payload(payload_bits, watermark_config)
        print(f"\nEncoded Message:")
        print(f"   - Core Data: '{payload_bits}' ({payload_bit_length} bits)")
        print(f"   - ECC Method: {watermark_config.get('ecc_method')}")
        print(f"   - Final Message: '{message_to_embed}' ({len(message_to_embed)} bits)")
    except ValueError as e:
        print(f"Config Error: {e}")
        return
    
    # ============ Outer Loop: Multiple Rounds ============
    for round_idx in range(total_rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_idx + 1}/{total_rounds}")
        print(f"{'='*60}")
        
        total_start_time = time.time()
        
        # Initialize log file
        initialize_log_file(
            str(standard_log_path),
            round_idx,
            total_rounds,
            str(verbose_log_path) if verbose_log_path else None,
            enable_verbose_log
        )
        
        # Execute single round
        all_responses, statistics = run_single_experiment_round(
            client=client,
            config=config,
            role_config=role_config,
            model=model,
            probability_template=probability_template,
            behavior_template=behavior_template,
            BEHAVIOR_TYPES=BEHAVIOR_TYPES,
            seed=seed,
            bit_stream=bit_stream,
            epoch_num=epoch_num,
            message_to_embed=message_to_embed,
            watermark_config=watermark_config
        )
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        # ============ Output Statistics ============
        print_statistics(statistics, round_idx, total_duration, epoch_num)
        
        # ============ Write Logs ============
        log_summary(
            str(standard_log_path),
            statistics,
            round_idx,
            total_duration,
            epoch_num
        )
        
        for i, round_data in enumerate(all_responses["rounds"], 1):
            log_round_results(str(standard_log_path), i, round_data)
        
        log_long_responses(
            str(verbose_log_path) if verbose_log_path else None,
            all_responses["long_responses"],
            enable_verbose_log
        )
        
        # Delay between rounds
        if round_idx < total_rounds - 1:
            print(f"\nResting for 5 seconds between rounds...")
            time.sleep(5)
    
    print(f"\n{'='*60}")
    print("All experiment rounds completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
