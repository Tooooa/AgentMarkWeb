"""
自定义少量 Agent 运行示例（带水印）

配置方式:
1. 复制 config.json.template 为 config.json
2. 填入你的 API 配置
3. 运行此脚本
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig

import oasis
from oasis import ActionType, AgentGraph, LLMAction, SocialAgent, UserInfo
from oasis.watermark import WatermarkManager


# ========== 从配置文件加载 ==========
def load_config(config_path: str = None) -> dict:
    """
    加载配置文件
    优先级: 指定路径 > ./config.json > ../config.json > 默认配置
    """
    # 尝试的配置文件路径列表
    search_paths = [
        config_path,
        "./config.json",
        "../config.json",
        str(Path(__file__).parent.parent / "config.json"),
    ]
    
    for path in search_paths:
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"✅ 已加载配置文件: {path}")
                return config
            except Exception as e:
                print(f"⚠️  配置文件 {path} 加载失败: {e}")
    
    # 返回默认配置
    print("⚠️  未找到配置文件，使用默认配置")
    return {
        "api_provider": "deepseek",
        "deepseek": {
            "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat"
        },
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini"
        },
        "num_agents": 2,
        "num_rounds": 3,
        "watermark_enabled": True,
        "watermark_config": {
            "payload_bit_length": 8,
            "ecc_method": "parity",
            "embedding_strategy": "cyclic"
        },
        "log_dir": "./log",
        "database_path": "./simulation.db"
    }


# 加载配置
CONFIG = load_config()
# ==============================


async def run_custom_simulation():
    """运行自定义配置的模拟"""
    
    # 读取配置
    api_provider = CONFIG.get("api_provider", "deepseek")
    num_agents = CONFIG.get("num_agents", 2)
    num_rounds = CONFIG.get("num_rounds", 3)
    use_deepseek = api_provider == "deepseek"
    
    print("=" * 70)
    print(f"🚀 OASIS 少量 Agent 模拟（带水印）")
    print(f"   Agents: {num_agents}")
    print(f"   Rounds: {num_rounds}")
    print(f"   LLM: {api_provider.upper()}")
    print("=" * 70)
    
    # 1. 配置 API
    if use_deepseek:
        deepseek_config = CONFIG.get("deepseek", {})
        os.environ["OPENAI_API_KEY"] = deepseek_config.get("api_key", "")
        os.environ["OPENAI_API_BASE"] = deepseek_config.get("base_url", "https://api.deepseek.com")
        print(f"\n✅ DeepSeek API 已配置")
        print(f"   Base URL: {deepseek_config.get('base_url')}")
    else:
        openai_config = CONFIG.get("openai", {})
        if openai_config.get("api_key"):
            os.environ["OPENAI_API_KEY"] = openai_config.get("api_key")
        print(f"\n✅ OpenAI API 已配置")
    
    # 2. ✅ 独立Agent架构：不再创建共享水印管理器
    # 每个Agent将自动创建自己的独立WatermarkManager，嵌入自己的agent_id
    print("\n📋 使用独立Agent水印架构...")
    print(f"   ✅ 每个Agent将自动创建独立的WatermarkManager")
    print(f"   📊 每个Agent嵌入自己的agent_id (8-bit binary)")
    watermark_config = CONFIG.get("watermark_config", {})
    log_dir = CONFIG.get("log_dir", "./log")
    
    # 3. 创建模型
    print("\n📋 创建 LLM 模型...")
    if use_deepseek:
        deepseek_config = CONFIG.get("deepseek", {})
        model_config = ChatGPTConfig(
            temperature=0.7, 
            max_tokens=1000,
        )
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=deepseek_config.get("model", "deepseek-chat"),
            model_config_dict=model_config.as_dict(),
            url=deepseek_config.get("base_url", "https://api.deepseek.com"),
            api_key=deepseek_config.get("api_key", ""),
        )
        print(f"   ✅ DeepSeek 模型: {deepseek_config.get('model', 'deepseek-chat')}")
    else:
        openai_config = CONFIG.get("openai", {})
        model_config = ChatGPTConfig(
            temperature=0.7,
            max_tokens=1000,
        )
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=openai_config.get("model", ModelType.GPT_4O_MINI),
            model_config_dict=model_config.as_dict(),
        )
    print(f"   ✅ 模型创建成功")
    
    # 4. 定义可用行为
    available_actions = [
        ActionType.LIKE_POST,
        ActionType.UNLIKE_POST,
        ActionType.DISLIKE_POST,
        ActionType.CREATE_POST,
        ActionType.CREATE_COMMENT,
        ActionType.REPOST,
        ActionType.QUOTE_POST,
        ActionType.FOLLOW,
        ActionType.UNFOLLOW,
        ActionType.MUTE,
        ActionType.UNMUTE,
        ActionType.SEARCH_USER,
        ActionType.SEARCH_POSTS,
        ActionType.REFRESH,
    ]
    print(f"\n📋 可用行为: {[a.value for a in available_actions]}")
    print(f"   ✅ 共 {len(available_actions)} 种行为可供选择")
    
    # 5. 创建 Agent Graph
    print(f"\n📋 创建 {num_agents} 个 Agent...")
    agent_graph = AgentGraph()
    agents = []
    
    for i in range(num_agents):
        agent = SocialAgent(
            agent_id=i,
            user_info=UserInfo(
                user_name=f"agent_{i}",
                name=f"Agent {i}",
                description=f"Social agent {i}",
                profile=None,
                recsys_type="reddit",
            ),
            agent_graph=agent_graph,
            model=model,
            available_actions=available_actions,
            # ✅ 不再传递watermark_manager，让Agent自动创建独立的
        )
        agent_graph.add_agent(agent)
        
        # ✅ 保存数字索引用于后续验证
        agent.agent_index = i  # 添加一个属性保存整数索引
        agents.append(agent)
        
        # 验证是否成功自动创建
        if hasattr(agent, 'watermark_manager') and agent.watermark_manager is not None:
            expected_bits = format(i, '08b')
            print(f"   ✅ Agent {i} 创建成功（独立水印，嵌入agent_id={i}, binary={expected_bits}）")
        else:
            print(f"   ⚠️ Agent {i} 未启用水印")
    
    # 6. 初始化环境
    print(f"\n📋 初始化 OASIS 环境...")
    db_path = CONFIG.get("database_path", f"./oasis_custom_{num_agents}agents_{num_rounds}rounds.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
    )
    await env.reset()
    print(f"   ✅ 环境初始化完成: {db_path}")
    
    # 7. 估算资源
    print(f"\n📊 资源估算:")
    total_calls = num_agents * num_rounds * 2  # 两阶段集成
    estimated_tokens = total_calls * 800
    if use_deepseek:
        estimated_cost = estimated_tokens * 0.00000025  # DeepSeek 定价
        print(f"   API 调用: {total_calls} 次")
        print(f"   估算 Tokens: ~{estimated_tokens:,}")
        print(f"   估算成本: ${estimated_cost:.4f} (DeepSeek)")
        print(f"   估算时间: {num_agents * num_rounds * 2} 秒 (约 {(num_agents * num_rounds * 2) / 60:.1f} 分钟)")
    else:
        estimated_cost = estimated_tokens * 0.000002  # OpenAI GPT-4O-MINI
        print(f"   API 调用: {total_calls} 次")
        print(f"   估算 Tokens: ~{estimated_tokens:,}")
        print(f"   估算成本: ${estimated_cost:.4f} (OpenAI)")
        print(f"   估算时间: {num_agents * num_rounds * 2} 秒 (约 {(num_agents * num_rounds * 2) / 60:.1f} 分钟)")
    
    # 8. 运行模拟
    print(f"\n" + "=" * 70)
    print(f"🎬 开始模拟 {num_rounds} 轮...")
    print("=" * 70)
    
    import time
    start_time = time.time()
    
    for round_num in range(num_rounds):
        print(f"\n📍 Round {round_num + 1}/{num_rounds}")
        
        # 所有 Agent 执行 LLM 驱动的行为
        all_actions = {agent: LLMAction() for agent in agents}
        
        try:
            round_start = time.time()
            await env.step(all_actions)
            round_time = time.time() - round_start
            
            # ✅ 显示每个Agent的独立统计
            print(f"   ✅ 完成 (耗时: {round_time:.1f}秒)")
            for agent in agents:
                if hasattr(agent, 'watermark_manager') and agent.watermark_manager is not None:
                    stats = agent.watermark_manager.get_statistics()
                    print(f"      Agent {agent.agent_index}: {stats['current_bit_index']}/{stats['bit_stream_length']} bits (剩余: {stats['bits_remaining']})")
            
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            break
    
    total_time = time.time() - start_time
    
    # 9. ✅ 独立提取和验证每个Agent的水印
    print(f"\n" + "=" * 70)
    print(f"🔍 提取和验证每个Agent的独立水印...")
    print("=" * 70)
    
    for agent in agents:
        if not (hasattr(agent, 'watermark_manager') and agent.watermark_manager is not None):
            print(f"\n⚠️ Agent {agent.agent_index}: 未启用水印")
            continue
        
        wm = agent.watermark_manager
        print(f"\n{'=' * 70}")
        print(f"🤖 Agent {agent.agent_index} - 水印提取")
        print(f"{'=' * 70}")
        
        extracted, stats = wm.extract_watermark_from_log()
        
        print(f"\n📊 提取结果:")
        print(f"   Agent索引(整数): {agent.agent_index}")
        print(f"   Agent UUID: {agent.social_agent_id}")
        print(f"   原始比特流: {wm.bit_stream} (长度: {len(wm.bit_stream)})")
        print(f"   提取比特流: {extracted} (长度: {len(extracted)})")
        print(f"   解码Payload: {stats.get('decoded_payload', 'N/A')}")
        
        # ✅ 识别agent_id：从解码的payload中提取前8位
        if stats.get('decoded_payload'):
            decoded_payload = stats.get('decoded_payload', '')
            if len(decoded_payload) >= 8:
                extracted_agent_id_bits = decoded_payload[:8]
                extracted_agent_id = int(extracted_agent_id_bits, 2)
                print(f"   识别的agent_id: {extracted_agent_id} (binary: {extracted_agent_id_bits})")
                
                if extracted_agent_id == agent.agent_index:
                    print(f"   ✅ Agent ID 匹配！")
                else:
                    print(f"   ❌ Agent ID 不匹配! (期望: {agent.agent_index})")
            else:
                print(f"   ⚠️ Payload不足8位，无法识别agent_id")
        
        # ✅ 更准确的统计描述
        total_rounds = stats.get('actions_processed', 0)
        embedded_rounds = stats.get('successful_extractions', 0)
        skipped_rounds = total_rounds - embedded_rounds
        
        print(f"\n📊 嵌入统计:")
        print(f"   总轮数: {total_rounds}")
        print(f"   有效嵌入轮数: {embedded_rounds} (成功嵌入水印)")
        if skipped_rounds > 0:
            print(f"   跳过轮数: {skipped_rounds} (概率分布太集中，无法嵌入)")
        print(f"   完整块数: {stats.get('complete_messages', 0)}")
        if stats.get('partial_bits', 0) > 0:
            print(f"   部分嵌入: {stats.get('partial_bits', 0)} bits {'✅ ECC验证通过' if stats.get('partial_is_valid') else '⚠️ ECC验证失败(不足完整块)'}")
        
        # ✅ 使用改进的准确率（来自stats）
        if len(extracted) > 0:
            # 优先使用 WatermarkManager 计算的循环准确率
            accuracy = stats.get('accuracy', 0.0)
            original_length = len(wm.bit_stream)
            extracted_length = len(extracted)
            
            print(f"\n📈 比特位准确度:")
            print(f"   - 匹配度: {accuracy:.1f}%")
            print(f"   - 原始长度: {original_length} bits")
            print(f"   - 提取长度: {extracted_length} bits")
            
            if accuracy < 100:
                # 显示不匹配的位置
                mismatches = []
                for i, bit in enumerate(extracted):
                    expected_bit = wm.bit_stream[i % len(wm.bit_stream)]
                    if bit != expected_bit:
                        mismatches.append((i, bit, expected_bit))
                
                if mismatches:
                    print(f"   ⚠️ 发现 {len(mismatches)} 个不匹配的bit:")
                    for pos, actual, expected in mismatches[:5]:  # 最多显示前5个
                        print(f"      位置{pos}: 提取='{actual}' vs 原始='{expected}'")
                    if len(mismatches) > 5:
                        print(f"      ... 还有 {len(mismatches) - 5} 个不匹配")
                    print(f"      可能原因: 概率分布波动导致解码误差")
            else:
                print(f"   ✅ 完美匹配: 提取的每一位都与原始bit_stream循环一致")
            
            print(f"\n🔐 ECC验证状态:")
            if stats.get('valid', False):
                print(f"   - 状态: ✅ 完全有效")
                print(f"   - 说明: 所有块通过ECC校验")
            else:
                print(f"   - 状态: ⚠️ 部分失败")
                print(f"   - 说明: {stats.get('failed_validations', 0)} 块ECC验证失败")
                if stats.get('partial_bits', 0) > 0:
                    print(f"   - 原因: 提取的 {stats.get('partial_bits', 0)} bits 不足完整块({stats.get('complete_messages', 0) + 1} * 9 = {(stats.get('complete_messages', 0) + 1) * 9} bits)")
            
            if accuracy == 100 and stats.get('valid', False):
                print(f"\n✅ Agent {agent.agent_index} 水印完整提取并验证成功！")
            elif accuracy >= 90:
                print(f"\n✅ Agent {agent.agent_index} 比特位准确度高（{accuracy:.1f}%）")
                if stats.get('partial_bits', 0) > 0:
                    print(f"   💡 提示: 增加模拟轮数可嵌入完整水印块，通过ECC验证")
            elif accuracy >= 80:
                print(f"\n⚠️ Agent {agent.agent_index} 水印提取准确率中等（{accuracy:.1f}%）")
                print(f"   💡 提示: 增加轮数以提高准确率")
            else:
                print(f"\n⚠️ Agent {agent.agent_index} 水印提取准确率较低（{accuracy:.1f}%）")
        else:
            print(f"\n❌ Agent {agent.agent_index} 未提取到水印")
    
    # 10. 清理
    await env.close()
    
    # 11. 总结
    print(f"\n" + "=" * 70)
    print(f"📊 运行总结")
    print("=" * 70)
    print(f"✅ 配置:")
    print(f"   - Agent 数量: {num_agents}")
    print(f"   - 模拟轮数: {num_rounds}")
    print(f"   - LLM: {api_provider.upper()}")
    print(f"\n✅ 性能:")
    print(f"   - 总耗时: {total_time:.1f} 秒 ({total_time / 60:.1f} 分钟)")
    print(f"   - 平均每轮: {total_time / num_rounds:.1f} 秒")
    print(f"   - API 调用: {total_calls} 次")
    print(f"\n✅ 输出文件:")
    print(f"   - 数据库: {db_path}")
    
    # ✅ 显示每个agent的日志文件
    for agent in agents:
        if hasattr(agent, 'watermark_manager') and agent.watermark_manager is not None:
            print(f"   - Agent {agent.agent_index} 日志: {agent.watermark_manager.log_file}")
    
    print("=" * 70)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     OASIS 自定义 Agent 运行（带水印）                    ║
    ║                                                          ║
    ║  修改配置:                                               ║
    ║    - NUM_AGENTS: Agent 数量                              ║
    ║    - NUM_ROUNDS: 模拟轮数                                ║
    ║    - USE_DEEPSEEK: 是否使用 DeepSeek                     ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(run_custom_simulation())
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
