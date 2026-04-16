import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_recall_fscore_support,
    ConfusionMatrixDisplay  # Added to match second script's plotting
)
from torch.utils.data import Dataset

# Extra cleaning for safety, since the model was trained on cleaned data
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())

# Label mapping function to unify labels into binary format
def map_label(label):
    label = str(label).lower().strip()
    if label in ["fake", "rumor", "conspiracy", "junksci"]:
        return 0
    if label in ["reliable", "political"]:
        return 1
    return None

# Dataset class for Hugging Face Trainer
class TokenizedDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Tokenization helper to handle large datasets without memory issues
def batched_tokenize(texts, tokenizer, batch_size=32):
    encodings = {"input_ids": [], "attention_mask": []}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokenized = tokenizer(
            batch,
            truncation=True,
            max_length=256,
            padding=False 
        )
        encodings["input_ids"].extend(tokenized["input_ids"])
        encodings["attention_mask"].extend(tokenized["attention_mask"])
    return encodings

# Load model and tokenizer
model_path = "data/distilbert_fake_news_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load data and splits
data_file = "data/news_cleaned_2018_02_13_cleaned_20pct.csv"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Could not find {data_file}")

df = pd.read_csv(data_file)
df["content"] = df["content"].apply(clean_text)
df["binary_label"] = df["type"].apply(map_label)
df = df.dropna(subset=["binary_label"])
df = df[df["content"].str.len() > 5]

X = df["content"].values
y = df["binary_label"].values.astype(int)

# Recreating the exact split used in training
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Test dataset preparation
test_encodings = batched_tokenize(list(X_test), tokenizer)
test_dataset = TokenizedDataset(test_encodings, y_test)

# Evaluation setup using Hugging Face Trainer for consistency with training
training_args = TrainingArguments(
    output_dir="./temp_eval",
    per_device_eval_batch_size=16,
    remove_unused_columns=False,
    report_to="none" 
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

print("\n--- Running Predictions ---")
preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=1)

# Metrics Calculation
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

print("\n" + "="*30)
print("CORE PERFORMANCE METRICS")
print("="*30)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*30)

print("\n=== FULL CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, digits=4, target_names=["Fake", "Real"]))

print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plotting the confusion matrix using sklearn's built-in functionality for better visualization
# Create the display object using the labels and CM data
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, values_format="d") 
plt.title("Confusion Matrix")
plt.tight_layout()

output_path = "Confusion_Matrices/DistilBERT/DistilBERT.png"

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

plt.savefig(output_path, dpi=300)
plt.show()