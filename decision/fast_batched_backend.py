import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class FastBatchedBackend:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.3", temperature=0.5):
        self.model_id = model_id
        self.temperature = temperature
        self.device = "cuda"

    def load(self):
        print(f"Loading {self.model_id} for BATCHED Inference...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("Backend loaded successfully.")

    def generate_batch(self, prompts: list[str], batch_size: int = 16) -> list[str]:
        """Processa dezenas de agentes na GPU simultaneamente."""
        all_responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150, 
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            for j, output in enumerate(outputs):
                input_len = inputs.input_ids[j].shape[0]
                new_tokens = output[input_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                all_responses.append(text)
                
        return all_responses