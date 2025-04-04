import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class InferenceModel_Generate:
    
    def __init__(self):
        self.MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
        self.MODEL_DIR = "./Model/Model_generate"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, cache_dir=self.MODEL_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.MODEL_DIR
        )
        
        print("Model is on", self.model.device)
    
    def inference(self, messages: list) -> str:
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.7,
                top_p=0.8
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response