import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import evaluate
import numpy as np
from config import Config

df = pd.read_csv(Config.test_file)
dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(Config.save_dir)
model = AutoModelForSequenceClassification.from_pretrained(Config.save_dir)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=Config.max_length)

dataset = dataset.map(tokenize)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels)["f1"]
    }

trainer = Trainer(model=model)
predictions = trainer.predict(dataset)

metrics = compute_metrics((predictions.predictions, dataset["label"]))

print("\nEvaluation Results:")
print(metrics)
