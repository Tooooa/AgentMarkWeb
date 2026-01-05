"""
ALFWorld prompt generator module.
Responsibilities: generate ALFWorld-specific prompt templates, including dynamic
ReAct few-shot examples and action blueprints.
"""

import json
import os
from collections import defaultdict
from typing import List, Dict

# Path setup: this file is under new_code/modules/alfworld/prompt.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Prefer local copy; fallback to ReAct directory if missing
LOCAL_PROMPT_FILE = os.path.join(os.path.dirname(__file__), 'alfworld_3prompts.json')
REACT_PROMPT_FILE = LOCAL_PROMPT_FILE if os.path.exists(LOCAL_PROMPT_FILE) else os.path.join(PROJECT_ROOT, 'ReAct', 'prompts', 'alfworld_3prompts.json')
with open(REACT_PROMPT_FILE, 'r', encoding='utf-8') as f:
    _react_prompts_raw = json.load(f)

REACT_PROMPT_GROUPS: Dict[str, List[str]] = defaultdict(list)
FEW_SHOT_EXAMPLES: List[Dict] = []
for key, prompt in _react_prompts_raw.items():
    lower = key.lower()
    if 'clean' in lower:
        category = 'clean'
    elif 'heat' in lower:
        category = 'heat'
    elif 'cool' in lower:
        category = 'cool'
    elif 'look' in lower:
        category = 'look'
    elif 'two' in lower:
        category = 'two_obj'
    else:
        category = 'put'
    REACT_PROMPT_GROUPS[category].append(prompt.strip())

ACTION_BLUEPRINTS = [
    {
        "title": "pick_and_place_simple (place)",
        "steps": [
            "search cabinet/drawer/countertop locations to find the target",
            "take <target>",
            "go to <destination>",
            "move <target> to <destination>"
        ]
    },
    {
        "title": "pick_clean_then_place (clean then place)",
        "steps": [
            "find the target and take",
            "go to sinkbasin 1",
            "clean <target> with sinkbasin 1",
            "go to <destination>",
            "move <target> to <destination>"
        ]
    },
    {
        "title": "pick_heat_then_place (heat then place)",
        "steps": [
            "find the target and take",
            "go to microwave 1",
            "heat <target> with microwave 1",
            "go to <destination>",
            "move <target> to <destination>"
        ]
    },
    {
        "title": "pick_cool_then_place (cool then place)",
        "steps": [
            "find the target and take",
            "go to fridge 1",
            "cool <target> with fridge 1",
            "go to <destination>",
            "move <target> to <destination>"
        ]
    },
    {
        "title": "pick_two_obj_and_place (two objects)",
        "steps": [
            "take first target -> go to <destination> -> move <target1> to <destination>",
            "return to origin -> take second target -> go to <destination> -> move <target2> to <destination>"
        ]
    },
    {
        "title": "look_at_obj_in_light (lamp inspection)",
        "steps": [
            "find the target and take (e.g., bowl/cd)",
            "go to the location where the desklamp is observed",
            "use desklamp (no need to put the item down)",
            "examine <target>"
        ]
    }
]


def _infer_task_category(task_description: str) -> str:
    text = (task_description or '').lower()
    if 'clean' in text:
        return 'clean'
    if 'heat' in text or 'warm' in text or 'hot' in text:
        return 'heat'
    if 'cool' in text or 'cold' in text:
        return 'cool'
    if 'light' in text or 'lamp' in text or 'examine' in text:
        return 'look'
    if 'two' in text or 'both' in text:
        return 'two_obj'
    return 'put'


def _select_react_prompts(task_description: str) -> List[str]:
    category = _infer_task_category(task_description)
    prompts = REACT_PROMPT_GROUPS.get(category)
    if prompts:
        return prompts
    merged = []
    for values in REACT_PROMPT_GROUPS.values():
        merged.extend(values)
    return merged


def format_react_examples(examples: List[str], max_examples: int = 3) -> str:
    slices = examples[:max_examples]
    formatted = []
    for idx, example in enumerate(slices, 1):
        formatted.append(f"ReAct Example {idx}:\n{example}\n")
    return "\n".join(formatted)


def format_action_blueprints(blueprints: List[Dict]) -> str:
    formatted = []
    for bp in blueprints:
        steps_text = "\n".join([f"    {idx+1}. {step}" for idx, step in enumerate(bp["steps"])])
        formatted.append(f"{bp['title']}:\n{steps_text}\n")
    return "\n".join(formatted)


def generate_alfworld_probability_prompt(
    observation: str,
    admissible_commands: List[str],
    task_description: str = None,
    few_shot_examples: List[Dict] = None,
    num_few_shot: int = 3,
    interaction_history: List[Dict] = None,
    holding_item: str = None,
    processed_item_status: Dict[str, str] = None,
    include_reasoning: bool = False
) -> str:
    selected_examples = _select_react_prompts(task_description)

    prompt_parts = []
    prompt_parts.append("You are an expert household task agent. Analyze the current situation and assign probabilities to each action.")

    if selected_examples:
        prompt_parts.append("\n[REACT FEW-SHOT EXAMPLES]")
        prompt_parts.append(format_react_examples(selected_examples, max_examples=num_few_shot))

    if ACTION_BLUEPRINTS:
        prompt_parts.append("\n[SUCCESS ACTION BLUEPRINTS]")
        prompt_parts.append(format_action_blueprints(ACTION_BLUEPRINTS))

    if interaction_history and len(interaction_history) > 0:
        prompt_parts.append("\n[YOUR RECENT ACTIONS - With Your Thinking]")
        recent_history = interaction_history[-5:] if len(interaction_history) > 5 else interaction_history
        for i, step in enumerate(recent_history, 1):
            thinking = step.get('thinking', '')
            if thinking:
                if len(thinking) > 150:
                    thinking = thinking[:150] + '...'
                prompt_parts.append(f"{i}. Your thinking: {thinking}")
            action = step['action']
            obs = step['observation']
            if len(obs) > 80:
                obs = obs[:80] + '...'
            prompt_parts.append(f"   Action: {action}")
            prompt_parts.append(f"   Result: {obs}")
        visited_locations = set()
        for step in interaction_history:
            if step['action'].startswith('go to '):
                visited_locations.add(step['action'][6:].strip())
        if visited_locations:
            prompt_parts.append(f"\nAlready visited: {', '.join(sorted(visited_locations))}")

    prompt_parts.append("\n[CURRENT SITUATION]")
    if holding_item:
        prompt_parts.append(f"YOUR INVENTORY: Holding {holding_item}")
        prompt_parts.append("   -> You cannot take another item until you put this down (use 'move' command)")
        if processed_item_status:
            state = processed_item_status.get(holding_item)
            if state:
                prompt_parts.append(f"   -> PROCESS STATUS: already {state.upper()} - focus on placement.")
    else:
        prompt_parts.append("YOUR INVENTORY: Empty (you can take items)")

    if processed_item_status:
        summary = "; ".join([f"{item}: {status}" for item, status in processed_item_status.items()])
        if summary:
            prompt_parts.append(f"PROCESS STATUS SUMMARY: {summary}")

    if task_description:
        prompt_parts.append(f"Task Goal: {task_description}")
        lower = task_description.lower()
        if 'clean' in lower:
            prompt_parts.append("  -> Processing: CLEAN at sinkbasin (take -> sinkbasin -> clean -> destination -> MOVE)")
        if 'heat' in lower or 'hot' in lower:
            prompt_parts.append("  -> Processing: HEAT in microwave (take -> microwave -> heat -> destination -> MOVE)")
        if 'cool' in lower or 'cold' in lower:
            prompt_parts.append("  -> Processing: COOL in fridge (take -> fridge -> cool -> destination -> MOVE)")
        if 'two' in lower or 'both' in lower:
            prompt_parts.append("  -> Special: handle TWO items sequentially (one at a time)")

    prompt_parts.append(f"Observation: {observation}")
    prompt_parts.append(f"Available Actions ({len(admissible_commands)} options):\n{json.dumps(admissible_commands, indent=2)}")

    prompt_parts.append("\n[YOUR RESPONSE FORMAT]")
    if include_reasoning:
        prompt_parts.append("First, write a 'Thinking: ...' section to analyze the situation.")
    else:
        prompt_parts.append("Write a Brief Analysis (1-3 sentences).")
        
    prompt_parts.append("Then output the JSON probability object (sum to 1.0).")

    return "\n".join(prompt_parts)


def format_commands_list(commands: List[str]) -> str:
    return "\n".join([f"  {i+1}. {cmd}" for i, cmd in enumerate(commands)])


def generate_example_json(commands: List[str]) -> str:
    example_dict = {cmd: f"<probability for '{cmd}'>" for cmd in commands}
    return json.dumps(example_dict, indent=2)


def generate_alfworld_action_prompt(
    observation: str,
    admissible_commands: List[str],
    task_description: str = None
) -> str:
    prompt_parts = []
    prompt_parts.append("You are an AI assistant helping to complete household tasks in a text-based environment.")
    if task_description:
        prompt_parts.append(f"Task: {task_description}")
    prompt_parts.append(f"Observation: {observation}")
    prompt_parts.append("\nAvailable Actions:")
    for idx, cmd in enumerate(admissible_commands, 1):
        prompt_parts.append(f"{idx}. {cmd}")
    prompt_parts.append("\nPlease choose the best action by responding with ONLY the action text.")
    return "\n".join(prompt_parts)
