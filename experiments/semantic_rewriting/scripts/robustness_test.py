
import os
import sys
import json
import random
import re
import math
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
from copy import deepcopy

# Add project root to sys.path
# Add project root to sys.path
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from openai import OpenAI
from agentmark.core.watermark_sampler import (
    sample_behavior_differential,
    sample_behavior,
    generate_contextual_key
)
from agentmark.core.parser_utils import extract_and_normalize_probabilities

# --- Configuration ---
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
TEMPERATURE = 1.0

if not API_KEY:
    print("Warning: DEEPSEEK_API_KEY not set. API calls will fail.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)



# Output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_dir = os.getenv("AGENT_MARK_OUTPUT_DIR", "./output")
default_output = os.path.join(base_output_dir, f"semantic_rewriting_{timestamp}")

def rewrite_observation(text: str) -> str:
    """Uses LLM to paraphrase the observation text while preserving semantics."""
    if not text.strip():
        return text

    sys_prompt = "You are a helpful assistant. Rewrite the following text to convey exactly the same information but using different words and sentence structures. Do not add or remove any facts. Keep it concise."
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text}
            ],
            temperature=1.0, # Use same temp or 1.0.
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Rewriting Error: {e}")
        return text

def apply_perturbation_to_prompt(prompt: str) -> str:
    """
    Identifies [CURRENT SITUATION] block and rewrites it using LLM.
    Preserves 'Available Actions'.
    """
    marker = "【CURRENT SITUATION】"
    if marker not in prompt:
        print("Warning: Marker not found, perturbing whole prompt")
        return rewrite_observation(prompt)
    
    parts = prompt.split(marker)
    pre_marker = parts[0]
    
    # The 'post' part contains the situation text + response format
    # We look for the NEXT section marker "【YOUR RESPONSE FORMAT】"
    
    current_situation_block_full = parts[1]
    
    end_marker = "【YOUR RESPONSE FORMAT】"
    if end_marker in current_situation_block_full:
        situation_content, suffix = current_situation_block_full.split(end_marker, 1)
        suffix = end_marker + suffix
    else:
        situation_content = current_situation_block_full
        suffix = ""
        
    # Inside situation_content, find "Available Actions"
    # Pattern: Available Actions (N options):\n[...]
    action_match = re.search(r'(Available Actions.*?:)(\s*\[.*?\])', situation_content, re.DOTALL)
    
    if action_match:
        header = action_match.group(1)
        json_list_str = action_match.group(2)
        start_idx = action_match.start()
        
        obs_text = situation_content[:start_idx]
        actions_text = header + json_list_str + situation_content[action_match.end():]
    else:
        obs_text = situation_content
        actions_text = ""
        
    # Perturb (Rewrite) the observation part
    # Remove markers like "✓ YOUR INVENTORY: ..." if checking strict observation?
    # The prompt includes "✓ YOUR INVENTORY: ... \nTask Goal: ... \nObservation: ..."
    # We should rewrite the "Observation:" part primarily, or the whole block?
    # "Current Situation" usually includes Inventory, Goal, Observation.
    # Let's rewrite the whole text PRECEDING actions to be safe, or just the "Observation" line?
    # The user said "remove 10% of text in CURRENT SITUATION".
    # Rewriting the whole block (Inventory + Goal + Obs) seems consistent with "Current Situation".
    
    print(f"Rewriting text of length {len(obs_text)}...")
    rewritten_obs = rewrite_observation(obs_text)
    print(f"\n[EXAMPLE REWRITE]\nORIGINAL:\n{obs_text}\n\nREWRITTEN:\n{rewritten_obs}\n[END EXAMPLE]\n")
    
    # Reassemble
    return f"{pre_marker}{marker}{rewritten_obs}\n{actions_text}{suffix}"

def extract_admissible_commands_from_prompt(prompt: str) -> List[str]:
    """Extracts the list of available actions from the prompt."""
    try:
        match = re.search(r'Available Actions.*?:(\s*\[.*?\])', prompt, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
    except Exception as e:
        print(f"Error parse available actions: {e}")
    return []

def get_llm_response(prompt: str, admissible_commands: List[str]) -> Dict[str, float]:
    """Call DeepSeek API and return probability dictionary using robust parser."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=256,
            top_p=1.0
        )
        content = response.choices[0].message.content
        # Use robust parser
        return extract_and_normalize_probabilities(content, admissible_commands)
    except Exception as e:
        print(f"API Error: {e}")
        return {}

def calculate_kl(p_p: Dict[str, float], p_q: Dict[str, float]) -> float:
    """KL Divergence(P || Q)"""
    # Normalize
    def normalize(p):
        total = sum(p.values())
        if total == 0: return p
        return {k: v/total for k, v in p.items()}
    
    p_p = normalize(p_p)
    p_q = normalize(p_q)
    
    # Align keys
    all_keys = set(p_p.keys()) | set(p_q.keys())
    epsilon = 1e-9
    
    kl = 0.0
    for k in all_keys:
        val_p = p_p.get(k, epsilon)
        val_q = p_q.get(k, epsilon)
        # Safe division
        if val_q <= 0:
             val_q = epsilon
        if val_p <= 0:
             continue # 0 * log(0) = 0 limit
             
        try:
             term = val_p * math.log(val_p / val_q)
             kl += term
        except ValueError:
             # math.log(non-positive)
             pass
        
    return kl

def run_experiment(task_log_path: str, decoded_bits_path: str, output_dir: str):
    # Load Logs
    with open(task_log_path, 'r') as f:
        task_log = json.load(f)
        
    with open(decoded_bits_path, 'r') as f:
        decoded_data = json.load(f)
        
    task_id = task_log['task_id']
    
    # Find bit stream for this task
    bit_stream = None
    for item in decoded_data:
        if item['task_id'] == task_id:
            bit_stream = item['decoded_bit_stream']
            break
            
    if not bit_stream:
        print(f"Could not find bit stream for task {task_id}")
        return

    print(f"Running Task {task_id}, BitStream Len: {len(bit_stream)}")
    
    results = []
    bit_index = 0
    prompts = task_log['prompts']
    
    # Sort prompts by step_num
    prompts.sort(key=lambda x: x['step_num'])
    
    # Limit steps logic
    steps_limit = getattr(args, 'steps', 9999)
    start_step = getattr(args, 'start_step', 1)
    executed_steps = 0

    for i, step_data in enumerate(prompts):
        step_num = step_data['step_num']
        
        if step_num < start_step:
            continue

        if executed_steps >= steps_limit:
            break
            
        original_prompt = step_data['prompt']
        
        # Extract admissible commands
        admissible_commands = extract_admissible_commands_from_prompt(original_prompt)
        if not admissible_commands:
            print(f"Step {step_num}: Could not find admissible commands, skipping.")
            continue

        print(f"--- Step {step_num} ---")
        
        # 1. Regenerate Baseline (Original Prompt)
        print("Querying Original...")
        probs_orig = get_llm_response(original_prompt, admissible_commands)
        if not probs_orig:
            print("Skipping step due to API failure")
            continue
            
        # Determine context for key (Previous Action)
        context_for_key = ""
        if "【YOUR RECENT ACTIONS - With Your Thinking】" in original_prompt:
             recent_block = original_prompt.split("【YOUR RECENT ACTIONS - With Your Thinking】")[1].split("【CURRENT SITUATION】")[0]
             # Find last "Action: ..." line
             actions = re.findall(r"Action: (.*)", recent_block)
             if actions:
                 context_for_key = actions[-1].strip()
        
        # Watermark Baseline
        action_orig, _, bits_encoded_orig, _ = sample_behavior_differential(
            probabilities=probs_orig,
            bit_stream=bit_stream,
            bit_index=bit_index,
            context_for_key=context_for_key,
            round_num=step_num 
        )
        
        # 2. Perturb and Run (Semantic Rewrite)
        print("Querying Perturbed (Rewritten)...")
        perturbed_prompt = apply_perturbation_to_prompt(original_prompt)
        probs_pert = get_llm_response(perturbed_prompt, admissible_commands)
        if not probs_pert:
             probs_pert = {k: 1.0/len(admissible_commands) for k in admissible_commands}
             print("Perturbed query failed, using uniform dist.")

        # Watermark Perturbed
        action_pert, _, bits_encoded_pert, _ = sample_behavior_differential(
            probabilities=probs_pert,
            bit_stream=bit_stream,
            bit_index=bit_index,
            context_for_key=context_for_key, # Crucial: Use SAME context (Teacher Forcing)
            round_num=step_num
        )
        
        # 3. Metrics
        kl = calculate_kl(probs_orig, probs_pert)
        match = (action_orig == action_pert)
        
        print(f"  Action Match: {match} ({action_orig} vs {action_pert})")
        print(f"  KL: {kl:.4f}")
        print(f"  Bits Encoded: {bits_encoded_orig} vs {bits_encoded_pert}")
        
        results.append({
            "step": step_num,
            "kl": kl,
            "action_match": match,
            "action_orig": action_orig,
            "action_pert": action_pert,
            "bits_orig": bits_encoded_orig,
            "bits_pert": bits_encoded_pert
        })
        
        bit_index = (bit_index + bits_encoded_orig) % len(bit_stream)
        executed_steps += 1

    # Save Report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"report_{task_id}_part_{start_step}.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved report to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="data/001_task_0.json")
    parser.add_argument("--bits", default="data/decoded_bits.json")
    # Generate timestamped default output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.getenv("AGENT_MARK_OUTPUT_DIR", "./output")
    default_output = os.path.join(base_output_dir, f"semantic_rewriting_{timestamp}")
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--steps", type=int, default=1, help="Limit number of steps to test")
    parser.add_argument("--start_step", type=int, default=1, help="Step to start from")
    
    args = parser.parse_args()
    
    run_experiment(args.task, args.bits, args.output)
