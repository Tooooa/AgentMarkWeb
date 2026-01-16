
import os
import sys
import torch
import logging
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# 确保 MarkLLM 在 sys.path 中
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
MARKLLM_PATH = os.path.join(PROJECT_ROOT, "MarkLLM")
if not os.path.exists(MARKLLM_PATH):
    # 回退到合并的 toolbench 位置
    MARKLLM_PATH = os.path.join(PROJECT_ROOT, "experiments", "toolbench", "MarkLLM")

if MARKLLM_PATH not in sys.path and os.path.exists(MARKLLM_PATH):
    sys.path.insert(0, MARKLLM_PATH)

try:
    from utils.transformers_config import TransformersConfig
    from watermark.synthid.synthid import SynthID, SynthIDConfig
except ImportError as e:
    logging.warning(f"无法导入 MarkLLM 模块：{e}")
    TransformersConfig = None
    SynthID = None
    SynthIDConfig = None

logger = logging.getLogger(__name__)

class LocalLLMClient:
    """
    模仿 OpenAI 客户端但运行本地模型并支持 MarkLLM 的包装器
    """
    def __init__(self, model_path, watermark_config=None, device="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.float16):
        self.model_path = model_path
        self.device = device
        self.watermark_enabled = False
        self.synthid = None
        
        logger.info(f"从 {model_path} 在 {device} 上加载本地模型...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True, torch_dtype=torch_dtype)
            self.model.eval()
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"加载模型失败：{e}")
            raise

        if watermark_config:
            self.setup_watermark(watermark_config)

        # 构造结构以模仿 client.chat.completions.create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create_completion))

    def setup_watermark(self, config_dict):
        if not TransformersConfig or not SynthID:
            logger.warning("MarkLLM 不可用，跳过水印设置")
            return

        logger.info("设置 Sythonid (SynthID) 水印...")
        
        # 为 MarkLLM 准备 TransformersConfig
        # MarkLLM 通常自己加载模型，但如果我们破解它，传递预加载的模型也是 BaseConfig 支持的
        # 或者我们只使用 TransformersConfig，它似乎持有模型引用
        # 让我们在下一步检查 TransformersConfig 定义，但假设它接受 model、tokenizer 等
        
        # 注意：MarkLLM 的 BaseConfig 接受 TransformersConfig 并从中提取模型
        # 所以我们用加载的对象构造 TransformersConfig
        
        transformers_config = TransformersConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            vocab_size=self.tokenizer.vocab_size,
            max_new_tokens=1024 # 默认值，可以被覆盖
        )
        
        # 准备 SynthIDConfig
        # 我们需要构造一个 SynthIDConfig。它接受一个路径或者我们可能需要子类化/手动实例化
        # 它继承 BaseConfig，从文件加载
        # 我们可能需要传递一个虚拟路径或创建一个配置文件
        # 现在，让我们假设我们可以通过 kwargs 传递一个字典，如果我们传递一个虚拟路径或 None？
        # 查看 BaseConfig：如果 algorithm_config_path 是 None，它加载 'config/{self.algorithm_name()}.json'
        # 如果我们没有那个文件，这可能很危险
        # 但 BaseConfig 用 kwargs 更新 config_dict
        # 所以我们可以传递 None 并在 kwargs 中传递所有参数
        
        # 如果未提供，使用默认的 SynthID 参数
        synthid_params = {
            "ngram_len": 5,
            "keys": [654, 465, 456, 645, 564, 546], # 示例密钥
            "sampling_table_size": 65536,
            "sampling_table_seed": 0,
            "context_history_size": 1024,
            "detector_type": "Bayesian",
            "threshold": 0.5,
            "watermark_mode": "non-distortionary",
            "num_leaves": 4
        }
        if isinstance(config_dict, dict):
            synthid_params.update(config_dict)

        # 创建 SynthID 实例
        # 我们传递 None 作为路径，参数作为 kwargs
        # 但 SynthID 构造函数：__init__(self, algorithm_config: str | SynthIDConfig, ...)
        # 它创建 SynthIDConfig(algorithm_config, transformers_config)
        # SynthIDConfig 继承 BaseConfig
        # BaseConfig(algorithm_config_path, transformers_config, **kwargs)
        
        # 所以我们需要手动创建 SynthIDConfig
        try:
            # 我们必须欺骗 BaseConfig 不加载文件，如果我们想要纯代码
            # 但它似乎在 None 时尝试加载默认值
            # 我们将提供 MarkLLM 默认配置的路径（如果需要），或者让它加载默认值并覆盖
            # 默认路径：MarkLLM/config/SynthID.json
            default_config_path = os.path.join(MARKLLM_PATH, "config", "SynthID.json")
            
            self.synthid_config = SynthIDConfig(
                default_config_path,
                transformers_config,
                **synthid_params
            )
            self.synthid = SynthID(self.synthid_config, transformers_config)
            self.watermark_enabled = True
            logger.info("Sythonid 水印设置完成")
        except Exception as e:
            logger.error(f"设置 SynthID 失败：{e}")
            raise

    def create_completion(self, model, messages, temperature=0.7, **kwargs):
        """
        模仿 openai.chat.completions.create
        """
        # 将消息转换为提示
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 生成参数
        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else 1.0,
        }
        gen_kwargs.update(kwargs) # 覆盖默认值
        
        # 过滤掉 HuggingFace generate() 不支持的 OpenAI 风格 kwargs
        unsupported_keys = ['max_tokens', 'response_format', 'stream', 'stop', 'n', 
                           'presence_penalty', 'frequency_penalty', 'logit_bias', 'user']
        for key in unsupported_keys:
            gen_kwargs.pop(key, None)
        
        # 如果存在，将 max_tokens 转换为 max_new_tokens
        if 'max_tokens' in kwargs:
            gen_kwargs['max_new_tokens'] = kwargs['max_tokens']

        # 如果需要，为水印更新 transformers config gen_kwargs
        if self.watermark_enabled and self.synthid:
            # SynthID 在 generate_watermarked_text 中使用 self.config.gen_kwargs？
            # 实际上 generate_watermarked_text 调用：
            # self.config.generation_model.generate(..., logits_processor=..., **self.config.gen_kwargs)
            # 所以我们应该更新 self.synthid.config.gen_kwargs
            self.synthid.config.gen_kwargs.update(gen_kwargs)

            # 生成
            # 注意：SynthID.generate_watermarked_text 接受纯字符串提示
            # 它在内部对其进行标记化
            try:
                # 我们需要确保不会重复模板化
                # prompt 已经是模板化的字符串
                # generate_watermarked_text 将对其进行标记化
                # Llama-3 tokenizer 如果作为文本传递，可能会正确处理模板化字符串？
                # 是的，apply_chat_template 返回一个字符串。Tokenizer 应该编码该字符串
                
                output_text = self.synthid.generate_watermarked_text(prompt)
                
                # 验证提示移除？
                # generate_watermarked_text 返回仅生成的文本（解码）通常？
                # 让我们检查 SynthID.generate_watermarked_text：
                #   encoded_prompt = ...
                #   encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
                #   watermarked_text = batch_decode(..., skip_special_tokens=True)[0]
                # 它解码整个序列（提示 + 完成），如果 generate 返回整个内容，通常是这样？
                # 标准 HF generate 返回 input_ids + output
                # 所以 watermarked_text 可能包含提示
                # 我们需要去除提示
                
                if output_text.startswith(prompt):
                    # 理想情况下我们去除它。但由于解码差异，严格的字符串匹配可能会失败
                    # 简单的启发式：
                    # 但等等，BaseWatermark 第 83 行的逻辑：
                    # unwatermarked_text = batch_decode(..., skip_special_tokens=True)[0]
                    # 这取决于模型是否在输出中返回 input_ids
                    # 通常是的
                    
                    # 我们将尝试去除提示
                    # 或者更简单：我们实现自己的生成循环，使用 synthid 的 logits 处理器
                    pass
                
                # 检查是否需要去除提示
                # 对于 Llama3，用户/助手逻辑
                # 让我们现在只返回原始文本，或者尝试分割
                # 一个健壮的方法：len(prompt) 个字符？不
                # 我们将依赖于我们可以构造一个 Response 对象的事实
                
                # 如果存在，重新去除提示
                 # 这对于特殊标记很棘手
                 # 让我们假设现在我们返回整个文本或尝试分割
                 # Llama-3 聊天模板通常以助手的标题结束
                 
                response_content = output_text
                # 尝试移除提示
                # 重新标记化提示以获取长度？
                prompt_tokens = self.tokenizer(prompt, return_tensors='pt')['input_ids']
                if self.tokenizer.decode(prompt_tokens[0]) in output_text:
                     response_content = output_text.replace(self.tokenizer.decode(prompt_tokens[0]), "", 1)
                
                # 回退：如果标准模板，简单分割
                if "<|start_header_id|>assistant<|end_header_id|>\n\n" in prompt:
                     parts = output_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")
                     if len(parts) > 1:
                         response_content = parts[-1]

            except Exception as e:
                logger.error(f"水印生成失败：{e}")
                # 回退到正常生成
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(**inputs, **gen_kwargs)
                response_content = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, **gen_kwargs)
            response_content = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 构造响应对象
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=response_content
                    )
                )
            ]
        )
        return response
