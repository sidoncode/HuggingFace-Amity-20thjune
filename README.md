# Hugging Face — A Simple Step-by-Step Demo

This guide walks you from zero to a working machine-learning demo in about 15 minutes. No deep ML knowledge required — if you can run a Python script, you can do this.

By the end you'll have:
- Run your first AI model with 3 lines of code
- Tried several ready-made tasks (sentiment, translation, summarization)
- Loaded a specific model from the Hugging Face Hub
- Built a tiny web app for your model

---

## What is Hugging Face?

Hugging Face is a platform that hosts thousands of free, pre-trained AI models (for text, images, audio, and more) plus the open-source `transformers` library that makes them easy to use. Think of it as "GitHub for machine-learning models."

The key idea: instead of training your own model, you download one someone already trained and use it in a few lines of code.

---

## Step 1 — Set up your environment

You need **Python 3.8 or newer**. Check your version:

```bash
python --version
```

It's good practice to create a virtual environment so this project's packages stay isolated:

```bash
# Create the environment
python -m venv hf-demo

# Activate it
# macOS / Linux:
source hf-demo/bin/activate
# Windows:
hf-demo\Scripts\activate
```

You'll know it worked when your terminal prompt shows `(hf-demo)`.

---

## Step 2 — Install the libraries

```bash
pip install transformers torch
```

- `transformers` — the Hugging Face library
- `torch` — PyTorch, the engine that runs the models

> **Tip:** The first install pulls in a few hundred MB. That's normal — it includes the deep-learning framework.

---

## Step 3 — Your first model (3 lines)

Create a file called `demo.py` and paste this in:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love learning new things with Hugging Face!")
print(result)
```

Run it:

```bash
python demo.py
```

**Expected output:**

```
[{'label': 'POSITIVE', 'score': 0.9998}]
```

That's it — you just ran a real AI model. 🎉

> **What happened?** The first time you run this, it downloads a default sentiment model and caches it on your machine. Later runs are instant because the model is already saved.

The `pipeline()` function is the magic here. You hand it a **task name** and it handles all the messy details (downloading the model, preparing your text, decoding the answer) for you.

---

## Step 4 — Try other tasks

Each task below is a one-line change. Swap them into your `demo.py` and re-run.

### Translation (English → French)

```python
from transformers import pipeline

translator = pipeline("translation_en_to_fr")
print(translator("Hugging Face makes machine learning simple."))
```

### Summarization

```python
from transformers import pipeline

summarizer = pipeline("summarization")
text = """
Hugging Face is a company and community that builds tools for machine
learning. Its most popular project is the transformers library, which gives
developers easy access to thousands of pre-trained models for tasks like text
classification, translation, and image recognition, all hosted on a central hub.
"""
print(summarizer(text, max_length=40, min_length=10))
```

### Fill in the blank

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
print(unmasker("Hugging Face is the best place to find AI <mask>."))
```

### Zero-shot classification (sort text into your own categories)

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
    "I need to return these shoes, they don't fit.",
    candidate_labels=["billing", "returns", "technical support"],
)
print(result)
```

Other task names worth trying: `"question-answering"`, `"text-generation"`, `"ner"` (named-entity recognition), and `"image-classification"`.

---

## Step 5 — Pick a specific model from the Hub

So far you've used each task's *default* model. But the whole point of Hugging Face is choice — you can browse models at **https://huggingface.co/models** and pick one by name.

Just pass `model=` to the pipeline:

```python
from transformers import pipeline

# A small, fast sentiment model fine-tuned on tweets
classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
)
print(classifier("This new update is absolutely amazing!"))
```

The string `"cardiffnlp/twitter-roberta-base-sentiment-latest"` is the model's ID on the Hub — always in the form `author/model-name`. Copy it straight from any model's page.

> **How to choose a model:** On the Models page, filter by task in the left sidebar, then sort by downloads or trending. Smaller models run faster; larger ones are usually more accurate.

---

## Step 6 — Turn it into a web app

Want a clickable demo instead of a script? Add **Gradio**, which builds a web interface in a few lines.

```bash
pip install gradio
```

Create `app.py`:

```python
import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def analyze(text):
    result = classifier(text)[0]
    return f"{result['label']} ({result['score']:.2%} confident)"

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Enter some text"),
    outputs=gr.Textbox(label="Sentiment"),
    title="Sentiment Analyzer",
    description="Type a sentence and see if it's positive or negative.",
)

demo.launch()
```

Run it:

```bash
python app.py
```

Open the local URL it prints (usually `http://127.0.0.1:7860`) in your browser. You now have a real interactive AI app.

> **Bonus:** You can publish this for free on **Hugging Face Spaces** (https://huggingface.co/spaces) so anyone can use it from a public link — no server setup needed.

---

## Common issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'transformers'` | Your virtual environment isn't active, or the install failed. Re-run Step 2. |
| First run is very slow | It's downloading the model. This only happens once per model. |
| Out-of-memory errors | Use a smaller model, or shorten your input text. |
| `torch` won't install | Visit https://pytorch.org for an install command matched to your system. |

---

## Where to go next

- **Browse models:** https://huggingface.co/models
- **Browse datasets:** https://huggingface.co/datasets
- **Official course (free):** https://huggingface.co/learn
- **Pipelines reference:** https://huggingface.co/docs/transformers/main_classes/pipelines

Once you're comfortable with pipelines, the natural next steps are loading models and tokenizers manually (for more control) and *fine-tuning* a model on your own data.

Happy building! 🤗
