"""
Reddit Watermark Agent Experiment Configuration
r/TechFuture Subreddit Simulation
"""

import os
import json
from pathlib import Path

# Project Root
REPO_ROOT = Path(__file__).resolve().parents[3]
OASIS_ROOT = REPO_ROOT / "experiments" / "oasis_watermark" / "oasis"
OUTPUT_ROOT = Path(os.getenv("OASIS_OUTPUT_ROOT", REPO_ROOT / "output" / "oasis"))

# ========== Load Project Configuration ==========
def load_project_config():
    """Load config.json from OASIS directory"""
    config_path = OASIS_ROOT / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load config file: {e}")
    return {}

_project_config = load_project_config()

# ========== Experiment Configuration ==========
EXPERIMENT_CONFIG = {
    "platform": "reddit",
    "num_steps": 10,                    # Number of time steps
    "num_watermark_agents": 5,          # Number of Watermark Agents
    "num_control_agents": 5,            # Number of Control Agents
    "output_root": str(OUTPUT_ROOT),
}

# ========== API Configuration (Priority: config.json > Env Vars) ==========
_deepseek_cfg = _project_config.get("deepseek", {})
API_CONFIG = {
    "provider": _project_config.get("api_provider", "deepseek"),
    "deepseek": {
        "api_key": _deepseek_cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": _deepseek_cfg.get("base_url", "https://api.deepseek.com"),
        "model": _deepseek_cfg.get("model", "deepseek-chat")
    },
    "openai": {
        "api_key": _project_config.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY", ""),
        "model": _project_config.get("openai", {}).get("model", "gpt-4o-mini")
    }
}

# ========== Watermark Configuration ==========
WATERMARK_CONFIG = {
    "enabled": True,
    "mode": "full",
    "ecc_method": "parity",
    "embedding_strategy": "cyclic",
    "delta": 0.1,
    "gamma": 0.3,
}

# ========== Available Reddit Actions ==========
AVAILABLE_ACTIONS = [
    "CREATE_POST",       # Create Post
    "CREATE_COMMENT",    # Create Comment
    "LIKE_POST",         # Like
    "DISLIKE_POST",      # Dislike
    "REFRESH",           # Refresh
    "DO_NOTHING",        # Do Nothing
]
