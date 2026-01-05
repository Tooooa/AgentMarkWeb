# -*- coding: utf-8 -*-
"""
Reddit 水印 Agent 实验模块
"""

from .config import EXPERIMENT_CONFIG, API_CONFIG, WATERMARK_CONFIG
from .personas import PERSONAS, get_persona_by_index, get_agent_name
from .seed_data import SEED_POSTS

__all__ = [
    "EXPERIMENT_CONFIG",
    "API_CONFIG", 
    "WATERMARK_CONFIG",
    "PERSONAS",
    "get_persona_by_index",
    "get_agent_name",
    "SEED_POSTS",
]
