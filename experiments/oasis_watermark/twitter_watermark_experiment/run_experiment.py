# -*- coding: utf-8 -*-
"""
Twitter Watermark Agent Experiment Script
Simulates a Twitter feed with 10 agents (5 Watermarked + 5 Control)
"""

import asyncio
import json
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parents[3]
OASIS_ROOT = REPO_ROOT / "experiments" / "oasis_watermark" / "oasis"
sys.path.insert(0, str(REPO_ROOT))
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# Local Imports
from config import EXPERIMENT_CONFIG, API_CONFIG
from personas import PERSONAS, get_persona_by_index, get_agent_name
from seed_data import SEED_TWEETS

OUTPUT_ROOT = Path(
    os.getenv(
        "OASIS_OUTPUT_ROOT",
        EXPERIMENT_CONFIG.get("output_root", REPO_ROOT / "output" / "oasis"),
    )
)
PLATFORM = EXPERIMENT_CONFIG.get("platform", "twitter")
OUTPUT_DIR = OUTPUT_ROOT / PLATFORM / f"{PLATFORM}_{RUN_ID}"
LOG_DIR = OUTPUT_DIR / "logs"
os.environ.setdefault("OASIS_LOG_DIR", str(LOG_DIR))
os.environ.setdefault("OASIS_WATERMARK_LOG_DIR", str(LOG_DIR / "watermark"))

# Add OASIS source path (use local modified version)
sys.path.insert(0, str(OASIS_ROOT))

# OASIS Imports
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.configs import ChatGPTConfig
import oasis
from oasis import ActionType, AgentGraph, LLMAction, ManualAction, SocialAgent, UserInfo
from oasis.watermark import WatermarkManager
from oasis.social_platform.typing import DefaultPlatformType
from oasis.social_platform.platform import Platform
from oasis.social_platform.channel import Channel

class TwitterWatermarkExperiment:
    """Twitter Watermark Agent Experiment"""
    
    def __init__(self):
        self.config = EXPERIMENT_CONFIG
        self.api_config = API_CONFIG
        
        self.watermark_agents: List[SocialAgent] = []
        self.control_agents: List[SocialAgent] = []
        self.all_agents: List[SocialAgent] = []
        
        self.agent_graph = None
        self.env = None
        self.model = None
        
        # Action logging
        self.action_history: Dict[int, List[Dict]] = {}
        self.agent_posts: Dict[int, List[int]] = {}
        
        # Output directory
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.watermark_log_dir = self.log_dir / "watermark"
        self.watermark_log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.output_dir / "simulation.db")

    def setup_api(self):
        """Configure LLM API"""
        print("\n" + "=" * 70)
        print("Configuring LLM API...")
        print("=" * 70)
        
        provider = self.api_config.get("provider", "deepseek")
        
        if provider == "deepseek":
            deepseek_cfg = self.api_config.get("deepseek", {})
            api_key = deepseek_cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY", "")
            base_url = deepseek_cfg.get("base_url", "https://api.deepseek.com")
            model_name = deepseek_cfg.get("model", "deepseek-chat")
            
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_API_BASE"] = base_url
            
            model_config = ChatGPTConfig(temperature=0.7, max_tokens=1000)
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=model_name,
                model_config_dict=model_config.as_dict(),
                url=base_url,
                api_key=api_key,
            )
            print(f"DeepSeek API Configured: {model_name}")
        else:
             print("Only DeepSeek is configured for this experiment (edit config.py for others).")
             sys.exit(1)

    def create_agents(self):
        """Create 10 Agents (5 Watermark + 5 Control) and build Social Graph"""
        print("\n" + "=" * 70)
        print("Creating Twitter Agents...")
        print("=" * 70)
        
        self.agent_graph = AgentGraph()
        # Use Twitter Actions
        available_actions = ActionType.get_default_twitter_actions()
        
        num_wm = self.config.get("num_watermark_agents", 5)
        num_ctrl = self.config.get("num_control_agents", 5)
        
        # 1. Create Agents
        # Watermark Group
        print("\nWatermark Group (Group A):")
        for i in range(num_wm):
            persona = get_persona_by_index(i)
            agent_name = get_agent_name(i, is_watermark=True)
            wm = WatermarkManager(
                enabled=True,
                mode="full",
                agent_id=i,
                log_dir=str(self.watermark_log_dir),
                config={
                    "payload_bit_length": 8,
                    "ecc_method": "parity",
                    "embedding_strategy": "cyclic",
                },
            )
            
            agent = SocialAgent(
                agent_id=i,
                user_info=UserInfo(
                    user_name=persona['user_name'],
                    name=agent_name,
                    description=persona['description'],
                    profile=persona['profile'],
                    recsys_type="twitter", # Important: recsys_type for agent
                ),
                agent_graph=self.agent_graph,
                model=self.model,
                available_actions=available_actions,
                watermark_manager=wm,
            )
            
            self.agent_graph.add_agent(agent)
            agent.agent_index = i
            agent.persona_name = persona['name']
            agent.is_watermark = True
            self.watermark_agents.append(agent)
            print(f"   Agent {i} [{persona['name']}] - Watermarked")

        # Control Group
        print("\nControl Group (Group B):")
        for i in range(num_ctrl):
            agent_id = num_wm + i
            persona = get_persona_by_index(i)
            agent_name = get_agent_name(agent_id, is_watermark=False)
            
            # Disable watermark
            disabled_wm = WatermarkManager(enabled=False)
            
            agent = SocialAgent(
                agent_id=agent_id,
                user_info=UserInfo(
                    user_name=f"ctrl_{persona['user_name']}",
                    name=agent_name,
                    description=persona['description'],
                    profile=persona['profile'],
                    recsys_type="twitter",
                ),
                agent_graph=self.agent_graph,
                model=self.model,
                available_actions=available_actions,
                watermark_manager=disabled_wm,
            )
            
            self.agent_graph.add_agent(agent)
            agent.agent_index = agent_id
            agent.persona_name = persona['name']
            agent.is_watermark = False
            self.control_agents.append(agent)
            print(f"   Agent {agent_id} [{persona['name']}] - Control")

        self.all_agents = self.watermark_agents + self.control_agents

        # 2. Build Follow Graph (Random Social Network)
        print("\nBuilding Follow Graph...")
        for agent in self.all_agents:
            # Randomly follow 3-5 other agents
            num_follows = random.randint(3, 5)
            potential_targets = [a for a in self.all_agents if a.agent_id != agent.agent_id]
            targets = random.sample(potential_targets, num_follows)
            
            for target in targets:
                self.agent_graph.add_edge(agent, target)
                # Note: We will also need to 'execute' these follows in the env initialization or manually
            
            print(f"   Agent {agent.agent_index} follows {len(targets)} agents.")

    async def setup_environment(self):
        """Initialize OASIS Twitter Environment"""
        print("\n" + "=" * 70)
        print("Initializing OASIS Environment...")
        print("=" * 70)
        
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        # Custom Platform Configuration to allow longer history
        channel = Channel()
        platform = Platform(
            db_path=self.db_path,
            channel=channel,
            recsys_type="twhin-bert", # Twitter Recommendation
            refresh_rec_post_count=5, # See 5 posts at a time
            max_rec_post_len=20,     # Keep 20 recent posts in buffer (User optimization)
            following_post_count=5,   # See 5 posts from following
        )

        self.env = oasis.make(
            agent_graph=self.agent_graph,
            platform=platform, # Use custom platform instance
            database_path=self.db_path,
        )
        
        await self.env.reset()
        print(f"Environment Ready: {self.db_path}")

    async def seed_initial_content(self):
        """Seed 20 initial tweets"""
        print("\n" + "=" * 70)
        print("Seeding Initial Content...")
        print("=" * 70)
        
        # We'll distribute seed tweets among random agents to simulate background noise
        # Or create a dummy 'Background' agent if we want strict separation.
        # For simplicity, let's have the agents themselves post the seed content 'historically'
        # or just random agents post them.
        
        for idx, tweet_data in enumerate(SEED_TWEETS):
            # Pick a random agent to be the 'author' of this history
            author = random.choice(self.all_agents)
            
            content = tweet_data['content']
            action = {
                author: ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": content}
                )
            }
            await self.env.step(action)
            
            # small delay
            if idx % 5 == 0:
                print(f"   Seeded {idx+1}/{len(SEED_TWEETS)} tweets...")
        
        print(f"Seeding Complete")

    async def run_simulation(self):
        """Run Main Simulation Loop"""
        num_steps = self.config.get("num_steps", 15)
        
        print("\n" + "=" * 70)
        print(f"Starting Simulation ({num_steps} Steps)...")
        print("=" * 70)
        
        start_time = time.time()
        
        for step in range(num_steps):
            print(f"\nStep {step + 1}/{num_steps}")
            step_start = time.time()
            
            # Execute Steps
            try:
                # 1. Refresh Feed first (so they see new content)
                refresh_actions = {agent: ManualAction(action_type=ActionType.REFRESH, action_args={}) for agent in self.all_agents}
                await self.env.step(refresh_actions)

                # 2. LLM Decide Action
                llm_actions = {agent: LLMAction() for agent in self.all_agents}
                await self.env.step(llm_actions)
                
                step_time = time.time() - step_start
                print(f"   Step Complete ({step_time:.1f}s)")
                
                # Check Watermark Stats
                for agent in self.watermark_agents:
                     if hasattr(agent, 'watermark_manager') and agent.watermark_manager:
                        stats = agent.watermark_manager.get_statistics()
                        print(f"      WM-{agent.agent_index}: {stats['current_bit_index']} bits")
                        
            except Exception as e:
                print(f"   Error: {e}")
                import traceback
                traceback.print_exc()

        total_time = time.time() - start_time
        print(f"\nSimulation Finished in {total_time:.1f}s")
    
    async def run(self):
        """Run Everything"""
        try:
            self.setup_api()
            self.create_agents()
            await self.setup_environment()
            await self.seed_initial_content()
            await self.run_simulation()
            
            # We will implement data export and evaluation in separate scripts/steps 
            # as per the prompt request order, but we can dump a basic json here.
            
            print("\n" + "=" * 70)
            print("Experiment Run Complete")
            print(f"   DB Path: {self.db_path}")
            print("=" * 70)

        except Exception as e:
            print(f"\nExperiment Failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
             if self.env:
                 await self.env.close()

async def main():
    experiment = TwitterWatermarkExperiment()
    await experiment.run()

if __name__ == "__main__":
    asyncio.run(main())
