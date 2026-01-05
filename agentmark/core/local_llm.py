
import os
import sys
import torch
import logging
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# Ensure MarkLLM is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
MARKLLM_PATH = os.path.join(PROJECT_ROOT, "MarkLLM")
if not os.path.exists(MARKLLM_PATH):
    # Fallback to the merged toolbench location
    MARKLLM_PATH = os.path.join(PROJECT_ROOT, "experiments", "toolbench", "MarkLLM")

if MARKLLM_PATH not in sys.path and os.path.exists(MARKLLM_PATH):
    sys.path.insert(0, MARKLLM_PATH)

try:
    from utils.transformers_config import TransformersConfig
    from watermark.synthid.synthid import SynthID, SynthIDConfig
except ImportError as e:
    logging.warning(f"Could not import MarkLLM modules: {e}")
    TransformersConfig = None
    SynthID = None
    SynthIDConfig = None

logger = logging.getLogger(__name__)

class LocalLLMClient:
    """
    A wrapper that mimics OpenAI client but runs a local model with MarkLLM support.
    """
    def __init__(self, model_path, watermark_config=None, device="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.float16):
        self.model_path = model_path
        self.device = device
        self.watermark_enabled = False
        self.synthid = None
        
        logger.info(f"Loading local model from {model_path} on {device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True, torch_dtype=torch_dtype)
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        if watermark_config:
            self.setup_watermark(watermark_config)

        # Structure to mimic client.chat.completions.create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create_completion))

    def setup_watermark(self, config_dict):
        if not TransformersConfig or not SynthID:
            logger.warning("MarkLLM not available, skipping watermark setup.")
            return

        logger.info("Setting up Sythonid (SynthID) watermark...")
        
        # Prepare TransformersConfig for MarkLLM
        # MarkLLM usually loads model itself, but passing pre-loaded model is supported by BaseConfig if we hack it,
        # OR we just use TransformersConfig which seems to hold model ref.
        # Let's check TransformersConfig definition in the next step, but assuming it takes model, tokenizer, etc.
        
        # NOTE: MarkLLM's BaseConfig takes TransformersConfig and PULLS model from it.
        # So we construct TransformersConfig with our loaded objects.
        
        transformers_config = TransformersConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            vocab_size=self.tokenizer.vocab_size,
            max_new_tokens=1024 # Default, can be overridden
        )
        
        # Prepare SynthIDConfig
        # We need to construct a SynthIDConfig. It takes a path or we might need to subclass/instantiate manually.
        # It inherits BaseConfig which loads from file. 
        # We might need to pass a dummy path or create a config file. 
        # For now, let's assume we can pass a dict via kwargs if we pass a dummy path or None?
        # Looking at BaseConfig: if algorithm_config_path is None, it loads 'config/{self.algorithm_name()}.json'.
        # That might be dangerous if we don't have that file.
        # But BaseConfig update config_dict with kwargs.
        # So we can pass None and pass all params in kwargs.
        
        # Default SynthID params if not provided
        synthid_params = {
            "ngram_len": 5,
            "keys": [654, 465, 456, 645, 564, 546], # Example keys
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

        # Create SynthID instance
        # We pass None as path, and params as kwargs.
        # But SynthID constructor: __init__(self, algorithm_config: str | SynthIDConfig, ...)
        # It creates SynthIDConfig(algorithm_config, transformers_config)
        # SynthIDConfig inherits BaseConfig.
        # BaseConfig(algorithm_config_path, transformers_config, **kwargs).
        
        # So we need to create SynthIDConfig manually.
        try:
            # We must trick BaseConfig to NOT load a file if we want to be pure code,
            # but it seems it attempts to load default if None.
            # We will provide the path to MarkLLM's default config if needed, or just let it load default and override.
            # Default path: MarkLLM/config/SynthID.json.
            default_config_path = os.path.join(MARKLLM_PATH, "config", "SynthID.json")
            
            self.synthid_config = SynthIDConfig(
                default_config_path,
                transformers_config,
                **synthid_params
            )
            self.synthid = SynthID(self.synthid_config, transformers_config)
            self.watermark_enabled = True
            logger.info("Sythonid watermark setup complete.")
        except Exception as e:
            logger.error(f"Failed to setup SynthID: {e}")
            raise

    def create_completion(self, model, messages, temperature=0.7, **kwargs):
        """
        Mimics openai.chat.completions.create
        """
        # Convert messages to prompt
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generation args
        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else 1.0,
        }
        gen_kwargs.update(kwargs) # Override defaults
        
        # Filter out OpenAI-style kwargs that HuggingFace generate() doesn't support
        unsupported_keys = ['max_tokens', 'response_format', 'stream', 'stop', 'n', 
                           'presence_penalty', 'frequency_penalty', 'logit_bias', 'user']
        for key in unsupported_keys:
            gen_kwargs.pop(key, None)
        
        # Convert max_tokens to max_new_tokens if present
        if 'max_tokens' in kwargs:
            gen_kwargs['max_new_tokens'] = kwargs['max_tokens']

        # Update transformers config gen_kwargs if needed for watermark
        if self.watermark_enabled and self.synthid:
            # SynthID uses self.config.gen_kwargs in generate_watermarked_text?
            # Actually generate_watermarked_text calls:
            # self.config.generation_model.generate(..., logits_processor=..., **self.config.gen_kwargs)
            # So we should update self.synthid.config.gen_kwargs
            self.synthid.config.gen_kwargs.update(gen_kwargs)

            # Generate
            # Note: SynthID.generate_watermarked_text takes a pure string prompt.
            # It tokenizes it internally.
            try:
                # We need to ensure we don't double-template.
                # prompt is already templated string.
                # generate_watermarked_text will tokenize it. 
                # Llama-3 tokenizer might handle templated string correctly if passed as text?
                # Yes, apply_chat_template returns a string. Tokenizer should encode that string.
                
                output_text = self.synthid.generate_watermarked_text(prompt)
                
                # Verify prompt removal?
                # generate_watermarked_text returns ONLY the generated text (decoded) usually?
                # Let's check SynthID.generate_watermarked_text:
                #   encoded_prompt = ...
                #   encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
                #   watermarked_text = batch_decode(..., skip_special_tokens=True)[0]
                # It decodes the WHOLE sequence (prompt + completion) usually if generate returns the whole thing?
                # Standard HF generate returns input + output.
                # So watermarked_text will likely contain the prompt.
                # We need to strip the prompt.
                
                if output_text.startswith(prompt):
                    # Ideally we strip it. But strict string matching might fail due to decoding diffs.
                    # Simple heuristic:
                    # But wait, logic in BaseWatermark line 83: 
                    # unwatermarked_text = batch_decode(..., skip_special_tokens=True)[0]
                    # It depends on if the model returns input_ids in output.
                    # Usually yes.
                    
                    # We will try to strip prompt.
                    # Or simpler: we implement our own generate loop using the logits processor from synthid.
                    pass
                
                # Check if we need to strip prompt.
                # For Llama3, user/assistant logic.
                # Let's just return raw text for now, or sophisticated stripping.
                # A robust way: len(prompt) chars? No.
                # We will rely on the fact that we can construct a Response object.
                
                # Re-strip prompt if present
                 # This is tricky with special tokens. 
                 # Let's assume for now we return the whole text or try to split.
                 # Llama-3 chat template usually ends with header for assistant.
                 
                response_content = output_text
                # Attempt to remove prompt
                # Re-tokenize prompt to get length?
                prompt_tokens = self.tokenizer(prompt, return_tensors='pt')['input_ids']
                if self.tokenizer.decode(prompt_tokens[0]) in output_text:
                     response_content = output_text.replace(self.tokenizer.decode(prompt_tokens[0]), "", 1)
                
                # Fallback: simple split if standard template
                if "<|start_header_id|>assistant<|end_header_id|>\n\n" in prompt:
                     parts = output_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")
                     if len(parts) > 1:
                         response_content = parts[-1]

            except Exception as e:
                logger.error(f"Watermark generation failed: {e}")
                # Fallback to normal generation
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(**inputs, **gen_kwargs)
                response_content = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, **gen_kwargs)
            response_content = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Construct response object
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
