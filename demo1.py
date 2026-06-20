from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I hate learning new things with Hugging Face!")
print(result)