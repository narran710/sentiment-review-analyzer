from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import Config

tokenizer = AutoTokenizer.from_pretrained(Config.save_dir)
model = AutoModelForSequenceClassification.from_pretrained(Config.save_dir)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
    return "Positive" if pred == 1 else "Negative"


# Example
if __name__ == "__main__":
    print(predict("The product was extremely good!"))
