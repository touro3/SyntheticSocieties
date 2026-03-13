import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Explain what a synthetic society is. Use clear, concise language."},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

text = tokenizer.decode(out[0], skip_special_tokens=True)

print("\n=== PROMPT (rendered) ===")
print(prompt)
print("\n=== OUTPUT ===")
print(text)