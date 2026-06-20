from transformers import pipeline

# A small, fast sentiment model fine-tuned on tweets
classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
)
print(classifier("This new update is absolutely amazing!"))