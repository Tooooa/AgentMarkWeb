# -*- coding: utf-8 -*-
"""
LLM-based Evaluation Script for Twitter Experiment
Evaluates agents on 5 dimensions using DeepSeek
"""

import os
import sys
import json
import sqlite3
import asyncio
import numpy as np
from pathlib import Path
from openai import AsyncOpenAI
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path(os.getenv("OASIS_OUTPUT_ROOT", REPO_ROOT / "output" / "oasis"))
sys.path.insert(0, str(REPO_ROOT))

# Load Config
from config import API_CONFIG, EXPERIMENT_CONFIG

# Configuration
DEEPSEEK_API_KEY = API_CONFIG["deepseek"]["api_key"] or os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = API_CONFIG["deepseek"]["base_url"]
DEEPSEEK_MODEL = API_CONFIG["deepseek"]["model"]

def _latest_run_dir(platform: str) -> Path:
    run_dir = os.getenv("OASIS_RUN_DIR")
    if run_dir:
        return Path(run_dir)
    candidates = sorted(
        [d for d in OUTPUT_ROOT.glob(f"{platform}_*") if d.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else OUTPUT_ROOT / f"{platform}_unknown"


OUTPUT_DIR = _latest_run_dir("twitter")
DATABASE_PATH = OUTPUT_DIR / "simulation.db"
if not DATABASE_PATH.exists():
    print(f"No experiment data found at {DATABASE_PATH}")
    sys.exit(1)

class AgentEvaluator:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        self.model = DEEPSEEK_MODEL

    def get_agents(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            # In OASIS, 'user' table stores agent info
            cursor.execute("SELECT * FROM user")
            agents = [dict(row) for row in cursor.fetchall()]
            return agents
        finally:
            conn.close()

    def get_agent_content(self, user_id: int, limit: int = 30) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        content_list = []
        try:
            cursor = conn.cursor()
            # Fetch Posts (Twitter uses post table for tweets, reposts, quotes)
            cursor.execute("SELECT content, quote_content FROM post WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit))
            posts = cursor.fetchall()
            
            for p in posts:
                text = p['content']
                if p['quote_content']:
                    text += f" [Quote: {p['quote_content']}]"
                if text:
                    content_list.append(text)
            
            return content_list
        finally:
            conn.close()

    async def evaluate_agent(self, agent: Dict[str, Any], content: List[str]) -> Dict[str, Any]:
        if not content:
            return {
                "user_id": agent.get('user_id'),
                "error": "No content"
            }

        content_str = "\n".join([f"- {c}" for c in content[:20]])
        profile = agent.get('bio', 'No profile available')
        name = agent.get('name', 'Unknown')

        prompt = f"""
Please evaluate the performance of the following Twitter AI Agent. Based on its Persona/Bio and recent Tweets, provide scores (1-10) for the following five dimensions.

Agent Name: {name}
Persona/Bio: {profile}

Recent Tweets:
{content_str}

Evaluation Metrics:
1. Logical Coherence: Is the content logically consistent?
2. Memory Accuracy: Are there any contradictions with previous statements? (If insufficient context to judge, give a high score)
3. Character Stability: Does the style and content align with the Profile/Bio description?
4. Social Norms & Common Sense: Does the behavior follow typical social media norms?
5. Language Diversity: Is the use of vocabulary, tags, and emojis rich and natural?

Please output strictly in JSON format without Markdown formatting (no ```json ... ```):
{{
    "logic_score": <int 1-10>,
    "memory_score": <int 1-10>,
    "stability_score": <int 1-10>,
    "norms_score": <int 1-10>,
    "diversity_score": <int 1-10>,
    "reason": "<short summary>"
}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert social media agent evaluator."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            result_json = response.choices[0].message.content
            return json.loads(result_json)
        except Exception as e:
            print(f"Error evaluating agent {name}: {e}")
            return {"error": str(e)}

    async def run(self):
        print(f"Reading Database: {self.db_path}")
        if not self.db_path.exists():
            print("Database file not found.")
            return

        agents = self.get_agents()
        print(f"   Found {len(agents)} agents.")
        
        results = []
        
        # We need to distinguish Watermark vs Control based on ID or Name
        # In our script: Watermark = 0-4, Control = 5-9.
        # Agent ID in DB matches loop index (0-9).
        
        print("\nStarting Evaluation with DeepSeek...")
        
        for agent in agents:
            user_id = agent.get('user_id')
            name = agent.get('name')
            
            # Determine Group
            # ID 0-4 (Watermark), 5-9 (Control)
            # note: user_id might be 0-indexed or equal to agent_id
            group = "Watermark" if user_id <= 4 else "Control"
            
            print(f"   Evaluating Agent {user_id} ({name}) [{group}]...")
            
            content = self.get_agent_content(user_id)
            if not content:
                print("      No content found.")
                continue
                
            eval_result = await self.evaluate_agent(agent, content)
            
            if "error" not in eval_result:
                complete_result = {
                    "user_id": user_id,
                    "name": name,
                    "group": group,
                    "metrics": eval_result
                }
                results.append(complete_result)
                print(f"      Score: L={eval_result['logic_score']}, S={eval_result['stability_score']}")
            else:
                print(f"      Evaluation Failed: {eval_result['error']}")

        # Save Results
        output_file = OUTPUT_DIR / "evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"\nEvaluation Saved: {output_file}")
        
        return results, output_file

if __name__ == "__main__":
    evaluator = AgentEvaluator(DATABASE_PATH)
    asyncio.run(evaluator.run())
