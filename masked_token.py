from transformers import pipeline

unmasker = pipeline(
    "fill-mask", model="Davlan/bert-base-multilingual-cased-finetuned-igbo"
)

result = unmasker(
    "Reno Omokri na Gọọmentị [MASK] enweghị ihe ha ga-eji hiwe ya bụ mmachi."
)

print(result)
