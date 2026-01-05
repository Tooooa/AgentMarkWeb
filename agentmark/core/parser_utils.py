"""
Parser utilities module.
Responsibilities: parse structured data from LLM responses.
"""

import re
import json


def extract_probabilities(response_text, behaviors):
    """
    Extract behavior probabilities from API response text.

    Supported strategies:
    1. Code block: parse ```json``` block (preferred)
    2. JSON parse: parse the first JSON object in the response
    3. Regex fallback: match each behavior probability with regex

    Extended support for ALFWorld command format:
    - Commands with special characters (e.g., "go to cabinet 1", "take soapbar 1 from countertop 1")
    - Partial probability extraction (returns partial dict)
    - Responses that include a Brief Analysis section

    Args:
        response_text (str): API response text
        behaviors (list): Behavior list

    Returns:
        dict: Probability dict, or None if extraction fails.
              Returns partial dict if only some probabilities are found.

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
    # Strategy 0: prefer JSON code block
    try:
        code_block_pattern = r"```json\s*\n([\s\S]*?)\n```"
        code_match = re.search(code_block_pattern, response_text)
        if code_match:
            json_text = code_match.group(1).strip()
            parsed = json.loads(json_text)
            
            # Check if all behavior keys exist
            if all(b in parsed for b in behaviors):
                return {b: float(parsed[b]) for b in behaviors}
            
            # Support partial extraction
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
        # Parsing failed, try next strategy
        pass
    
    # Strategy 1: parse the first JSON object (more robust)
    try:
        # Find first brace-wrapped JSON chunk
        m = re.search(r"\{[\s\S]*?\}", response_text)
        if m:
            json_text = m.group(0)
            # Models may use single quotes; try to make valid JSON
            json_text_fixed = json_text.replace("'", '"')
            parsed = json.loads(json_text_fixed)
            
            # Check if all behavior keys exist
            if all(b in parsed for b in behaviors):
                return {b: float(parsed[b]) for b in behaviors}
            
            # Support partial extraction (ALFWorld-specific)
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
        # Parsing failed, try regex fallback
        pass

    # Strategy 2: regex per behavior (supports partial matches)
    partial_result = {}
    for behavior in behaviors:
        # Escape special characters for ALFWorld commands
        escaped_behavior = re.escape(behavior)
        # Pattern: "behavior": 0.5 or 'behavior': 0.5
        pattern = r'["\']' + escaped_behavior + r'["\']\s*:\s*([0-9]*\.?[0-9]+)'
        match = re.search(pattern, response_text)
        if match:
            try:
                partial_result[behavior] = float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    if partial_result:
        return partial_result

    # Strategy 3: match full set of behaviors (legacy fallback)
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
    Extract and normalize probabilities from LLM response (multi-level fallback).

    Three-level fallback strategy:
    1. Strategy 1: extract all probabilities and normalize
    2. Strategy 2: partial normalization, set missing to 0
    3. Strategy 3: uniform distribution on total failure

    All fallback usage is logged.

    Args:
        response_text (str): LLM response text
        admissible_commands (list): Admissible commands
        logger (logging.Logger, optional): Logger for fallback usage

    Returns:
        dict: Normalized probability dict {command: probability}

    Example:
        >>> response = '{"go to cabinet 1": 0.3, "take soapbar 1": 0.7}'
        >>> commands = ["go to cabinet 1", "take soapbar 1", "open cabinet 1"]
        >>> extract_and_normalize_probabilities(response, commands)
        {'go to cabinet 1': 0.3, 'take soapbar 1': 0.7, 'open cabinet 1': 0.0}
    """
    # Try extracting probabilities
    extracted = extract_probabilities(response_text, admissible_commands)
    
    # Strategy 1: full extraction and normalization
    if extracted and len(extracted) == len(admissible_commands):
        total = sum(extracted.values())
        if total > 0:
            normalized = {k: v / total for k, v in extracted.items()}
            if logger:
                logger.debug(f"Extracted and normalized all {len(extracted)} probabilities")
            return normalized
        else:
            # All probabilities are 0; use uniform distribution
            if logger:
                logger.warning("Sum of extracted probabilities is 0; using uniform fallback")
            uniform_prob = 1.0 / len(admissible_commands)
            return {cmd: uniform_prob for cmd in admissible_commands}
    
    # Strategy 2: partial normalization, missing set to 0
    elif extracted and len(extracted) > 0:
        if logger:
            logger.warning(
                f"Only extracted {len(extracted)}/{len(admissible_commands)} probabilities; "
                "using partial normalization"
            )
        
        # Check actions not in admissible list (filtered out)
        filtered_actions = {k: v for k, v in extracted.items() if k not in admissible_commands}
        if filtered_actions and logger:
            logger.warning(f"LLM actions not in admissible list (filtered): {filtered_actions}")
        
        # Keep only admissible commands
        valid_extracted = {k: v for k, v in extracted.items() if k in admissible_commands}
        
        # Normalize valid extracted probabilities
        total = sum(valid_extracted.values())
        if total > 0:
            normalized = {k: v / total for k, v in valid_extracted.items()}
            if logger and filtered_actions:
                # Show before/after normalization
                logger.info(f"Before normalization (LLM raw): {extracted}")
                logger.info(f"After normalization (filtered+redistributed): {normalized}")
        else:
            # All extracted probabilities are 0; assign uniform probability
            uniform_prob = 1.0 / len(valid_extracted)
            normalized = {k: uniform_prob for k in valid_extracted.keys()}
        
        # Set missing commands to 0
        for cmd in admissible_commands:
            if cmd not in normalized:
                normalized[cmd] = 0.0
        
        if logger:
            logger.info(f"Extracted valid actions: {list(valid_extracted.keys())}")
            logger.info(f"Missing actions (set to 0): {[c for c in admissible_commands if c not in extracted]}")
        
        return normalized
    
    # Strategy 3: uniform distribution on total failure
    else:
        if logger:
            logger.error(
                f"Failed to extract any probabilities; using uniform fallback. "
                f"Response text: {response_text[:200]}..."
            )
        
        uniform_prob = 1.0 / len(admissible_commands)
        return {cmd: uniform_prob for cmd in admissible_commands}
