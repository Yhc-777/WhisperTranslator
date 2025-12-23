# whispertranslator/qwen3.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from pathlib import Path

@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_new_tokens: int = 2048
    presence_penalty: float = 0.0

@dataclass
class GenerationResult:
    text: str
    thinking_content: str = ""

class Qwen3:
    def __init__(self, model_path, device_map="auto", torch_dtype=torch.float16):
        """
        初始化 Qwen3 模型
        
        Args:
            model_path: 模型路径，可以是HuggingFace模型名称或本地路径
                       例如: "Qwen/Qwen3-0.6B" 或 "./models/Qwen3-0.6B"
            device_map: 设备映射，默认"auto"
            torch_dtype: 数据类型，默认float16
        """
        print(f"Loading Qwen3 model from: {model_path}")
        
        # 检查是否为本地路径
        if Path(model_path).exists():
            print(f"Loading from local path: {model_path}")
            local_files_only = True
        else:
            print(f"Loading from HuggingFace: {model_path}")
            local_files_only = False
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=local_files_only,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                local_files_only=local_files_only,
                trust_remote_code=True
            )
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
    def infer(self, system_prompt, user_input, gen_config: GenerationConfig, enable_thinking=False):
        """
        推理方法
        Args:
            system_prompt: 系统提示词
            user_input: 用户输入
            gen_config: 生成配置
            enable_thinking: 是否启用思考模式（翻译任务建议关闭以提高效率）
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 生成参数
        gen_kwargs = {
            "max_new_tokens": gen_config.max_new_tokens,
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "do_sample": True if gen_config.temperature > 0 else False,
        }
        
        # 如果使用vllm或其他支持presence_penalty的框架，可以添加
        if hasattr(gen_config, 'presence_penalty') and gen_config.presence_penalty > 0:
            gen_kwargs["repetition_penalty"] = 1.0 + (gen_config.presence_penalty / 10.0)
        
        generated_ids = self.model.generate(
            **model_inputs,
            **gen_kwargs
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # 解析思考内容（如果启用了thinking模式）
        thinking_content = ""
        if enable_thinking:
            try:
                # 查找 </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
                thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            except ValueError:
                # 没有找到thinking block
                content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        else:
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        return GenerationResult(text=content, thinking_content=thinking_content)
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
