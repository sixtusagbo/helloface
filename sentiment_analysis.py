from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("I'm very happy to use the Transformers library")

print(result)
