from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

model_name = "mbeukman/xlm-roberta-base-finetuned-igbo-finetuned-ner-igbo"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

nlp = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
)
example = "Chinua Achebe bụ onye edemede Nigeria"

ner_results = nlp(example)
print(ner_results)
