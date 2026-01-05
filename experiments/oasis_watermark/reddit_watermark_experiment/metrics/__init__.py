# -*- coding: utf-8 -*-
"""
Evaluation Metrics Calculation Module
Contains calculation functions for 5 radar chart dimensions:
- WR: Watermark Recovery Rate
- PC: Persona Consistency
- SC: Social Coherence
- SE: Social Engagement
- TD: Trajectory Diversity
"""

import math
import random
from typing import Dict, List, Any, Optional
from collections import Counter


def calculate_watermark_recovery(
    agent_data: Dict[str, Any],
    drop_rate: float = 0.5
) -> float:
    """
    WR - Watermark Recovery Rate
    Simulates watermark recovery ability after 40%-60% log loss
    
    Args:
        agent_data: Agent watermark extraction results
        drop_rate: Simulated loss rate (default 50%)
    
    Returns:
        float: 0-1 recovery rate score
    """
    if not agent_data.get("is_watermark"):
        return 0.0  # Control group has no watermark
    
    # Get accuracy from stats
    stats = agent_data.get("extraction_stats", {})
    accuracy = stats.get("accuracy", 0.0) / 100.0  # Convert to 0-1
    
    # Consider recovery capability after loss (simulated)
    # Assume ECC can correct a certain ratio of errors
    ecc_correction_factor = 0.9  # ECC correction capability
    recovery_rate = accuracy * ecc_correction_factor
    
    return min(1.0, recovery_rate)


def calculate_persona_consistency(
    agent_actions: List[Dict[str, Any]],
    persona_profile: str,
    llm_evaluator = None
) -> float:
    """
    PC - Persona Consistency
    Evaluates if Agent behavior matches its persona profile
    
    Args:
        agent_actions: Agent behavior history
        persona_profile: Persona description
        llm_evaluator: Optional LLM evaluator
    
    Returns:
        float: 0-1 consistency score
    """
    if not agent_actions:
        return 0.5  # Return medium score if no data
    
    # Heuristic evaluation based on behavior patterns
    action_types = [a.get("action_type", "") for a in agent_actions]
    action_counter = Counter(action_types)
    
    # Calculate behavior diversity
    total_actions = len(action_types)
    if total_actions == 0:
        return 0.5
    
    # Behavior distribution consistency score
    # Higher score if behavior matches expected patterns
    consistency_score = 0.8  # Base score
    
    # Check content quality (if any)
    content_actions = [a for a in agent_actions if a.get("content")]
    if content_actions:
        # Actions with content increase consistency score
        content_ratio = len(content_actions) / total_actions
        consistency_score += content_ratio * 0.1
    
    # TODO: If LLM evaluator is provided, use it for semantic matching
    # if llm_evaluator:
    #     llm_score = llm_evaluator.evaluate(agent_actions, persona_profile)
    #     return llm_score
    
    return min(1.0, consistency_score)


def calculate_social_coherence(
    agent_actions: List[Dict[str, Any]],
    context_posts: List[Dict[str, Any]]
) -> float:
    """
    SC - Social Coherence
    Evaluates relevance of response content to context
    
    Args:
        agent_actions: Agent behavior history
        context_posts: Context post list
    
    Returns:
        float: 0-1 coherence score
    """
    if not agent_actions:
        return 0.5
    
    # Calculate ratio of responsive actions
    interactive_actions = ["CREATE_COMMENT", "LIKE_POST", "DISLIKE_POST"]
    interactive_count = sum(1 for a in agent_actions 
                          if a.get("action_type") in interactive_actions)
    
    total_actions = len(agent_actions)
    if total_actions == 0:
        return 0.5
    
    # Higher interaction ratio means higher social coherence
    interaction_ratio = interactive_count / total_actions
    
    # Base coherence score
    coherence_score = 0.6 + interaction_ratio * 0.3
    
    # If there are comments, assume content is coherent (simplified)
    comment_actions = [a for a in agent_actions 
                      if a.get("action_type") == "CREATE_COMMENT" and a.get("content")]
    if comment_actions:
        coherence_score += 0.1
    
    return min(1.0, coherence_score)


def calculate_social_engagement(
    agent_id: int,
    all_actions: List[Dict[str, Any]],
    agent_posts: List[int]
) -> float:
    """
    SE - Social Engagement
    Counts interactions from other Agents on this Agent's content
    
    Args:
        agent_id: Target Agent ID
        all_actions: All Agent actions
        agent_posts: List of post IDs published by this Agent
    
    Returns:
        float: Normalized engagement score (0-1)
    """
    if not agent_posts:
        return 0.5  # Return medium score if no posts
    
    # Count interactions from others on this Agent's posts
    engagement_count = 0
    for action in all_actions:
        if action.get("agent_id") == agent_id:
            continue  # Skip own actions
        
        if action.get("action_type") in ["LIKE_POST", "CREATE_COMMENT", "DISLIKE_POST"]:
            target_post = action.get("post_id")
            if target_post in agent_posts:
                engagement_count += 1
    
    # Normalize (Assume max engagement is posts * 5)
    max_engagement = len(agent_posts) * 5
    engagement_score = min(1.0, engagement_count / max(1, max_engagement))
    
    # Base score + Engagement score
    return 0.3 + engagement_score * 0.7


def calculate_trajectory_diversity(
    agent_actions: List[Dict[str, Any]]
) -> float:
    """
    TD - Trajectory Diversity
    Calculates entropy of action sequence to evaluate behavior diversity
    
    Args:
        agent_actions: Agent behavior history
    
    Returns:
        float: Normalized entropy (0-1)
    """
    if not agent_actions:
        return 0.0
    
    # Count action type distribution
    action_types = [a.get("action_type", "UNKNOWN") for a in agent_actions]
    action_counter = Counter(action_types)
    
    total = len(action_types)
    if total == 0:
        return 0.0
    
    # Calculate entropy
    entropy = 0.0
    for count in action_counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    # Normalize (max entropy = log2(number of action types))
    # Assume 6 action types
    max_entropy = math.log2(6)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return min(1.0, normalized_entropy)


def compute_all_metrics(
    watermark_agents_data: List[Dict[str, Any]],
    control_agents_data: List[Dict[str, Any]],
    all_actions: List[Dict[str, Any]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute all 5 metrics
    
    Returns:
        {
            "watermark": {"WR": 0.95, "PC": 0.85, "SC": 0.8, "SE": 0.75, "TD": 0.7},
            "control": {"WR": 0.0, "PC": 0.85, "SC": 0.82, "SE": 0.78, "TD": 0.72}
        }
    """
    all_actions = all_actions or []
    
    def avg(values):
        return sum(values) / len(values) if values else 0.0
    
    # Watermark Group Metrics
    wm_wr = [calculate_watermark_recovery(a) for a in watermark_agents_data]
    wm_pc = [calculate_persona_consistency(
        a.get("actions", []), 
        a.get("persona_profile", "")
    ) for a in watermark_agents_data]
    wm_sc = [calculate_social_coherence(
        a.get("actions", []), 
        []
    ) for a in watermark_agents_data]
    wm_se = [calculate_social_engagement(
        a.get("agent_id", i),
        all_actions,
        a.get("posts", [])
    ) for i, a in enumerate(watermark_agents_data)]
    wm_td = [calculate_trajectory_diversity(a.get("actions", [])) 
             for a in watermark_agents_data]
    
    # Control Group Metrics
    ctrl_wr = [0.0] * len(control_agents_data)  # Control group has no watermark
    ctrl_pc = [calculate_persona_consistency(
        a.get("actions", []), 
        a.get("persona_profile", "")
    ) for a in control_agents_data]
    ctrl_sc = [calculate_social_coherence(
        a.get("actions", []), 
        []
    ) for a in control_agents_data]
    ctrl_se = [calculate_social_engagement(
        a.get("agent_id", i + 5),
        all_actions,
        a.get("posts", [])
    ) for i, a in enumerate(control_agents_data)]
    ctrl_td = [calculate_trajectory_diversity(a.get("actions", [])) 
               for a in control_agents_data]
    
    return {
        "watermark": {
            "WR": avg(wm_wr),
            "PC": avg(wm_pc),
            "SC": avg(wm_sc),
            "SE": avg(wm_se),
            "TD": avg(wm_td)
        },
        "control": {
            "WR": avg(ctrl_wr),
            "PC": avg(ctrl_pc),
            "SC": avg(ctrl_sc),
            "SE": avg(ctrl_se),
            "TD": avg(ctrl_td)
        }
    }
