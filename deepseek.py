"""
Tried to run Deepseek, but I don't have a compatible GPU🫠

I'll try other methods or a cloud option next.
Just wanna play around it.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-r1", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-r1")
prompt = "Explain quantum computing in simple terms."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)

print(len(outputs))
print(tokenizer.decode(outputs[0]))
