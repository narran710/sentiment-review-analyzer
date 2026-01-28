from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

MODEL_DIR = Path("models/bert_model")

app = FastAPI(title="Sentiment Review Analyzer API")

# Load model once (fast)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)


class InputText(BaseModel):
    text: str


@app.post("/predict")
def predict(input_data: InputText):
    text = input_data.text

    # Tokenize
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Predict
    with torch.no_grad():
        logits = model(**tokens).logits
        prediction = torch.argmax(logits, dim=1).item()

    label_map = {0: "negative", 1: "positive"}

    return {
        "text": text,
        "sentiment": label_map[prediction]
    }


@app.get("/")
def home():
    return {"status": "API running", "message": "Welcome to Sentiment Review Analyzer!"}
