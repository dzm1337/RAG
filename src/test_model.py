import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float32,
)

input_ids = tokenizer(
    "Plants create energy through a process known as", return_tensors="pt"
)

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0]))
