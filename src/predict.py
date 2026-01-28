import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/bert_model"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def predict(text: str):
    """Return sentiment prediction for given text."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label_id = torch.argmax(probs, dim=1).item()

    label = "Positive" if label_id == 1 else "Negative"
    confidence = probs[0][label_id].item()

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }

if __name__ == "__main__":
    text = input("Enter text to analyze: ")
    result = predict(text)
    print(f"Sentiment: {result['label']} (confidence: {result['confidence']})")
