"""
Train a sentiment analysis model using DistilBERT on the IMDB dataset.
Compatible with transformers 4.32.0 and accelerate.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import evaluate
import numpy as np
from pathlib import Path
from src.data_preprocessing import clean_text

# ----------------------
# CONFIG
# ----------------------
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/bert_model"
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5

# ----------------------
# Load dataset
# ----------------------
print("Loading IMDB dataset...")
raw = load_dataset("imdb")

# Simple preprocessing
def preprocess(examples):
    examples["text"] = [clean_text(t) for t in examples["text"]]
    return examples

raw = raw.map(preprocess, batched=True)

# ----------------------
# Tokenization
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

encoded = raw.map(tokenize, batched=True)
encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ----------------------
# Model
# ----------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# ----------------------
# Metrics
# ----------------------
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# ----------------------
# Training arguments
# ----------------------


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_dir="logs",
    logging_steps=50,
    save_total_limit=2,
    overwrite_output_dir=True,
    # remove evaluation_strategy, save_strategy if not supported
)

# ----------------------
# Trainer
# ----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"].shuffle(seed=42).select(range(10000)),  # quick demo
    eval_dataset=encoded["test"].select(range(2000)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ----------------------
# Train
# ----------------------
if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete! Model saved to:", OUTPUT_DIR)
