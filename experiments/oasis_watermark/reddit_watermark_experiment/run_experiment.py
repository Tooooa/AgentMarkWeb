# -*- coding: utf-8 -*-
"""
Reddit Watermark Agent Experiment Main Script
r/TechFuture Subreddit Simulation

Experiment Goals:
- Initialize 10 Agents (5 Watermark + 5 Control)
- Run 10 time steps in Reddit environment
- Calculate 5 evaluation metrics
- Generate radar chart comparison

Usage:
    cd oasis
    python examples/reddit_watermark_experiment/run_experiment.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parents[3]
OASIS_ROOT = REPO_ROOT / "experiments" / "oasis_watermark" / "oasis"
sys.path.insert(0, str(REPO_ROOT))
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# Import local modules (independent of OASIS)
from config import EXPERIMENT_CONFIG, API_CONFIG, WATERMARK_CONFIG
from personas import PERSONAS, get_persona_by_index, get_agent_name
from seed_data import SEED_POSTS
from metrics import compute_all_metrics

OUTPUT_ROOT = Path(
    os.getenv(
        "OASIS_OUTPUT_ROOT",
        EXPERIMENT_CONFIG.get("output_root", REPO_ROOT / "output" / "oasis"),
    )
)
PLATFORM = EXPERIMENT_CONFIG.get("platform", "reddit")
OUTPUT_DIR = OUTPUT_ROOT / PLATFORM / f"{PLATFORM}_{RUN_ID}"
LOG_DIR = OUTPUT_DIR / "logs"
os.environ.setdefault("OASIS_LOG_DIR", str(LOG_DIR))
os.environ.setdefault("OASIS_WATERMARK_LOG_DIR", str(LOG_DIR / "watermark"))

# Add OASIS source path (use local modified version)
sys.path.insert(0, str(OASIS_ROOT))

# Import OASIS modules
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig

import oasis
from oasis import ActionType, AgentGraph, LLMAction, ManualAction, SocialAgent, UserInfo
from oasis.watermark import WatermarkManager


class RedditWatermarkExperiment:
    """Reddit Watermark Agent Experiment Class"""
    
    def __init__(self):
        self.config = EXPERIMENT_CONFIG
        self.api_config = API_CONFIG
        self.watermark_config = WATERMARK_CONFIG
        
        self.watermark_agents: List[SocialAgent] = []
        self.control_agents: List[SocialAgent] = []
        self.all_agents: List[SocialAgent] = []
        
        self.agent_graph = None
        self.env = None
        self.model = None
        
        # Action history
        self.action_history: Dict[int, List[Dict]] = {}  # agent_id -> actions
        self.agent_posts: Dict[int, List[int]] = {}  # agent_id -> post_ids
        
        # Output directories
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.watermark_log_dir = self.log_dir / "watermark"
        self.watermark_log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / "simulation.db"
    
    def setup_api(self):
        """Configure API and Model"""
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
            openai_cfg = self.api_config.get("openai", {})
            api_key = openai_cfg.get("api_key") or os.getenv("OPENAI_API_KEY", "")
            model_name = openai_cfg.get("model", "gpt-4o-mini")
            
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            
            model_config = ChatGPTConfig(temperature=0.7, max_tokens=1000)
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=model_name,
                model_config_dict=model_config.as_dict(),
            )
            print(f"OpenAI API Configured: {model_name}")
    
    def create_agents(self):
        """Create 10 Agents (5 Watermark + 5 Control)"""
        print("\n" + "=" * 70)
        print("Creating Agents...")
        print("=" * 70)
        
        self.agent_graph = AgentGraph()
        available_actions = ActionType.get_default_reddit_actions()
        
        num_wm = self.config.get("num_watermark_agents", 5)
        num_ctrl = self.config.get("num_control_agents", 5)
        
        # ========== Create Watermark Agents ==========
        print("\nWatermark Group (Group A):")
        for i in range(num_wm):
            persona = get_persona_by_index(i)
            agent_name = get_agent_name(i, is_watermark=True)
            
            wm = WatermarkManager(
                enabled=self.watermark_config.get("enabled", True),
                mode=self.watermark_config.get("mode", "full"),
                agent_id=i,
                log_dir=str(self.watermark_log_dir),
                config={
                    "payload_bit_length": 8,
                    "ecc_method": self.watermark_config.get("ecc_method", "parity"),
                    "embedding_strategy": self.watermark_config.get("embedding_strategy", "cyclic"),
                },
            )

            # Watermark Group: Explicitly pass WatermarkManager, ensure logs write to independent output directory
            agent = SocialAgent(
                agent_id=i,
                user_info=UserInfo(
                    user_name=f"wm_{persona['user_name']}",
                    name=agent_name,
                    description=persona['description'],
                    profile=persona['profile'],
                    recsys_type="reddit",
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
            self.action_history[i] = []
            self.agent_posts[i] = []
            
            if hasattr(agent, 'watermark_manager') and agent.watermark_manager:
                expected_bits = format(i, '08b')
                print(f"   Agent {i} [{persona['name']}] - Watermark Enabled (agent_id={i}, bits={expected_bits})")
            else:
                print(f"   Agent {i} [{persona['name']}] - Watermark Disabled")
        
        # ========== Create Control Agents ==========
        print("\nControl Group (Group B):")
        for i in range(num_ctrl):
            agent_id = num_wm + i
            persona = get_persona_by_index(i)  # 使用相同人设
            agent_name = get_agent_name(agent_id, is_watermark=False)
            
            # Control Group: Pass disabled WatermarkManager
            disabled_wm = WatermarkManager(enabled=False)
            
            agent = SocialAgent(
                agent_id=agent_id,
                user_info=UserInfo(
                    user_name=f"ctrl_{persona['user_name']}",
                    name=agent_name,
                    description=persona['description'],
                    profile=persona['profile'],
                    recsys_type="reddit",
                ),
                agent_graph=self.agent_graph,
                model=self.model,
                available_actions=available_actions,
                watermark_manager=disabled_wm,  # Disable watermark
            )
            
            self.agent_graph.add_agent(agent)
            agent.agent_index = agent_id
            agent.persona_name = persona['name']
            agent.is_watermark = False
            self.control_agents.append(agent)
            self.action_history[agent_id] = []
            self.agent_posts[agent_id] = []
            
            print(f"   Agent {agent_id} [{persona['name']}] - No Watermark (Control Group)")
        
        self.all_agents = self.watermark_agents + self.control_agents
        print(f"\nTotal Agents Created: {len(self.all_agents)}")
    
    async def setup_environment(self):
        """Initialize OASIS Environment"""
        print("\n" + "=" * 70)
        print("Initializing OASIS Environment...")
        print("=" * 70)
        
        db_path = str(self.db_path)
        if os.path.exists(db_path):
            os.remove(db_path)
        
        self.env = oasis.make(
            agent_graph=self.agent_graph,
            platform=oasis.DefaultPlatformType.REDDIT,
            database_path=db_path,
        )
        await self.env.reset()
        print(f"Environment Initialized: {db_path}")
    
    async def seed_initial_content(self):
        """Post Seed Content and Initial Comments"""
        print("\n" + "=" * 70)
        print("Posting Seed Content (r/TechFuture)...")
        print("=" * 70)
        
        # Use first agent to post seed posts
        seed_agent = self.all_agents[0]
        
        for idx, post_data in enumerate(SEED_POSTS, start=1):
            content = f"【{post_data['title']}】\n\n{post_data['content']}"
            
            action = {
                seed_agent: ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": content}
                )
            }
            await self.env.step(action)
            print(f"   Seed Post {idx}: {post_data['title'][:30]}...")
            
            # Post initial comments (use different agents)
            for comment_idx, comment in enumerate(post_data.get("initial_comments", [])):
                comment_agent = self.all_agents[(idx + comment_idx) % len(self.all_agents)]
                comment_action = {
                    comment_agent: ManualAction(
                        action_type=ActionType.CREATE_COMMENT,
                        action_args={"post_id": idx, "content": comment["content"]}
                    )
                }
                await self.env.step(comment_action)
            
            await asyncio.sleep(0.2)
        
        print(f"Seed Content Posted: {len(SEED_POSTS)} posts")
    
    async def run_simulation(self):
        """Run Simulation"""
        num_steps = self.config.get("num_steps", 10)
        
        print("\n" + "=" * 70)
        print(f"Starting Simulation {num_steps} steps...")
        print("=" * 70)
        
        start_time = time.time()
        
        for step in range(num_steps):
            print(f"\n📍 Step {step + 1}/{num_steps}")
            step_start = time.time()
            
            # All Agents execute LLM-driven actions
            all_actions = {agent: LLMAction() for agent in self.all_agents}
            
            try:
                await self.env.step(all_actions)
                step_time = time.time() - step_start
                
                # Show progress
                print(f"   Completed (Time: {step_time:.1f}s)")
                
                # Show Watermark Group stats
                for agent in self.watermark_agents:
                    if hasattr(agent, 'watermark_manager') and agent.watermark_manager:
                        stats = agent.watermark_manager.get_statistics()
                        print(f"      WM-{agent.agent_index}: {stats['current_bit_index']}/{stats['bit_stream_length']} bits")
                
            except Exception as e:
                print(f"   Error: {e}")
                import traceback
                traceback.print_exc()
        
        total_time = time.time() - start_time
        print(f"\nSimulation Completed: Total Time {total_time:.1f}s ({total_time/60:.1f} min)")
    
    def collect_actions_from_db(self):
        """Collect all Agent actions from database"""
        import sqlite3
        
        db_path = str(self.db_path)
        
        print("\n" + "=" * 70)
        print("Collecting Action Data from Database...")
        print("=" * 70)
        
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get all posts
            cursor.execute("SELECT * FROM post")
            posts = cursor.fetchall()
            print(f"   Total Posts: {len(posts)}")
            
            # Get all comments
            cursor.execute("SELECT * FROM comment")
            comments = cursor.fetchall()
            print(f"   Total Comments: {len(comments)}")
            
            # Get all likes/dislikes
            cursor.execute("SELECT * FROM like")
            likes = cursor.fetchall()
            print(f"   Total Likes/Dislikes: {len(likes)}")
            
            # Collect actions for each Agent
            for agent in self.all_agents:
                agent_id = agent.agent_index
                user_id = agent_id + 1  # user_id starts from 1 in DB
                
                actions = []
                agent_post_ids = []
                
                # Collect post actions
                for post in posts:
                    if post['user_id'] == user_id:
                        actions.append({
                            "action_type": "CREATE_POST",
                            "agent_id": agent_id,
                            "post_id": post['post_id'],
                            "content": post['content'][:100] if post['content'] else "",
                            "created_at": post['created_at']
                        })
                        agent_post_ids.append(post['post_id'])
                
                # Collect comment actions
                for comment in comments:
                    if comment['user_id'] == user_id:
                        actions.append({
                            "action_type": "CREATE_COMMENT",
                            "agent_id": agent_id,
                            "post_id": comment['post_id'],
                            "content": comment['content'][:100] if comment['content'] else "",
                            "created_at": comment['created_at']
                        })
                
                # Collect like/dislike actions
                for like in likes:
                    if like['user_id'] == user_id:
                        action_type = "LIKE_POST" if like.get('like_type', 1) == 1 else "DISLIKE_POST"
                        actions.append({
                            "action_type": action_type,
                            "agent_id": agent_id,
                            "post_id": like['post_id'],
                            "created_at": like['created_at']
                        })
                
                self.action_history[agent_id] = actions
                self.agent_posts[agent_id] = agent_post_ids
                
                print(f"   Agent {agent_id}: {len(actions)} actions, {len(agent_post_ids)} posts")
            
            conn.close()
            print(f"Action collection complete")
            
        except Exception as e:
            print(f"Database read failed: {e}")
            import traceback
            traceback.print_exc()
    
    def extract_watermarks(self):
        """Extract and Verify Watermarks"""
        print("\n" + "=" * 70)
        print("Extracting Watermarks...")
        print("=" * 70)
        
        results = []
        
        for agent in self.watermark_agents:
            if not (hasattr(agent, 'watermark_manager') and agent.watermark_manager):
                continue
            
            wm = agent.watermark_manager
            extracted, stats = wm.extract_watermark_from_log()
            
            result = {
                "agent_id": agent.agent_index,
                "persona": agent.persona_name,
                "is_watermark": True,
                "original_bits": wm.bit_stream,
                "extracted_bits": extracted,
                "extraction_stats": stats,
                "actions": self.action_history.get(agent.agent_index, []),
                "posts": self.agent_posts.get(agent.agent_index, []),
                "persona_profile": PERSONAS[agent.agent_index % 5]['profile']
            }
            results.append(result)
            
            # Verify
            decoded = stats.get('decoded_payload', '')
            accuracy = stats.get('accuracy', 0)
            
            if decoded and len(decoded) >= 8:
                extracted_id = int(decoded[:8], 2)
                match = "MATCH" if extracted_id == agent.agent_index else "MISMATCH"
                print(f"   Agent {agent.agent_index} [{agent.persona_name}]: {match} Identified as {extracted_id} (Accuracy: {accuracy:.1f}%)")
            else:
                print(f"   Agent {agent.agent_index} [{agent.persona_name}]: Incomplete Extraction ({len(extracted)} bits)")
        
        return results
    
    def collect_control_data(self) -> List[Dict]:
        """Collect Control Group Data"""
        results = []
        for agent in self.control_agents:
            result = {
                "agent_id": agent.agent_index,
                "persona": agent.persona_name,
                "is_watermark": False,
                "actions": self.action_history.get(agent.agent_index, []),
                "posts": self.agent_posts.get(agent.agent_index, []),
                "persona_profile": PERSONAS[agent.agent_index % 5]['profile']
            }
            results.append(result)
        return results
    
    def compute_metrics(self, watermark_data: List[Dict], control_data: List[Dict]) -> Dict:
        """Compute Evaluation Metrics"""
        print("\n" + "=" * 70)
        print("Computing Metrics...")
        print("=" * 70)
        
        all_actions = []
        for agent_id, actions in self.action_history.items():
            all_actions.extend(actions)
        
        metrics = compute_all_metrics(watermark_data, control_data, all_actions)
        
        print("\nWatermark Group Metrics:")
        for k, v in metrics['watermark'].items():
            print(f"   {k}: {v:.3f}")
        
        print("\nControl Group Metrics:")
        for k, v in metrics['control'].items():
            print(f"   {k}: {v:.3f}")
        
        return metrics
    
    async def run(self):
        """Run Full Experiment"""
        print("\n" + "=" * 70)
        print("Reddit Watermark Agent Experiment")
        print("   r/TechFuture Subreddit Simulation")
        print("=" * 70)
        print(f"   Watermark Group: {self.config.get('num_watermark_agents', 5)} Agents")
        print(f"   Control Group: {self.config.get('num_control_agents', 5)} Agents")
        print(f"   Time Steps: {self.config.get('num_steps', 10)} Steps")
        print(f"   Output Directory: {self.output_dir}")
        print("=" * 70)
        
        try:
            # 1. Configure API
            self.setup_api()
            
            # 2. Create Agents
            self.create_agents()
            
            # 3. Initialize Environment
            await self.setup_environment()
            
            # 4. Post Seed Content
            await self.seed_initial_content()
            
            # 5. Run Simulation
            await self.run_simulation()
            
            # 6. Collect Actions from DB
            self.collect_actions_from_db()
            
            # 7. Extract Watermarks
            watermark_data = self.extract_watermarks()
            control_data = self.collect_control_data()
            
            # 8. Compute Metrics
            metrics = self.compute_metrics(watermark_data, control_data)
            
            # 8. Save Metrics
            metrics_path = self.output_dir / "metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"Metrics saved: {metrics_path}")
            
            # 9. Cleanup
            await self.env.close()
            
            print("\n" + "=" * 70)
            print("Experiment Completed!")
            print("=" * 70)
            print(f"   Metrics Data: {self.output_dir / 'metrics.json'}")
            print("=" * 70)
            
            return metrics
            
        except Exception as e:
            print(f"\nExperiment Failed: {e}")
            import traceback
            traceback.print_exc()
            raise


async def main():
    """Main Function"""
    experiment = RedditWatermarkExperiment()
    await experiment.run()


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     Reddit Watermark Agent Experiment                            ║
    ║     r/TechFuture Subreddit Simulation                            ║
    ║                                                                  ║
    ║     5 Watermark Agents + 5 Control Agents                        ║
    ║     Metrics: WR / PC / SC / SE / TD                              ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nUser Interrupted")
    except Exception as e:
        print(f"\n\nRun Failed: {e}")
        import traceback
        traceback.print_exc()
