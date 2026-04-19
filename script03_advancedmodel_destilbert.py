import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from torch.utils.data import Dataset

# Cleaning function, text is already cleaned in previous steps, but we can add extra cleaning for safety
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())


def is_valid_sample(text: str, label) -> bool:
    if text is None or len(text) < 5:
        return False
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return False
    return True


# Dataset
class TokenizedDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=0  # Focus on fake news metrics
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# Class weights for imbalanced datasets
def get_class_weights(labels):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    return torch.tensor(weights, dtype=torch.float)


# Custom loss trainer to handle class weights
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        if "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# Main
class FakeNewsTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )

    @staticmethod
    def map_label(label):
        label = str(label).lower().strip()
        if label in ["fake", "rumor", "conspiracy", "junksci"]:
            return 0
        if label in ["reliable", "political"]:
            return 1
        return None

    def _batched_tokenize(self, texts, batch_size=32):
        encodings = {"input_ids": [], "attention_mask": []}
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokenized = self.tokenizer(
                batch, truncation=True, max_length=256, padding=False 
            )
            encodings["input_ids"].extend(tokenized["input_ids"])
            encodings["attention_mask"].extend(tokenized["attention_mask"])
        return encodings

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df["content"] = df["content"].apply(clean_text)
        df["binary_label"] = df["type"].apply(self.map_label)
        df = df.dropna(subset=["binary_label"])
        df = df[df["content"].str.len() > 5]
        
        X = df["content"].values
        y = df["binary_label"].values.astype(int)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        self.train_dataset = TokenizedDataset(self._batched_tokenize(list(X_train)), y_train)
        self.val_dataset = TokenizedDataset(self._batched_tokenize(list(X_val)), y_val)
        self.test_dataset = TokenizedDataset(self._batched_tokenize(list(X_test)), y_test)
        self.class_weights = get_class_weights(y_train)

    def train(self):
        output_dir = "./results"
        training_args = TrainingArguments(
            output_dir=output_dir,
            save_strategy="epoch",
            save_total_limit=2,
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_strategy="epoch",
            fp16=torch.cuda.is_available(),
            # CHANGED: Set to 0 to prevent MemoryError on Windows with large datasets
            dataloader_num_workers=0, 
            remove_unused_columns=False,
        )

        self.trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer, 
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=compute_metrics,
            class_weights=self.class_weights,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        checkpoint = True if os.path.exists(output_dir) and os.listdir(output_dir) else None
        self.trainer.train(resume_from_checkpoint=checkpoint)

    def evaluate(self):
        print("\nVALIDATION:", self.trainer.evaluate(self.val_dataset))
        print("\nTEST:", self.trainer.evaluate(self.test_dataset))

    def save(self):
        path = "data/distilbert_fake_news_model"
        os.makedirs(path, exist_ok=True)
        self.trainer.save_model(path)
        self.tokenizer.save_pretrained(path)

    def run(self):
        self.load_data()
        self.train()
        self.evaluate()
        self.save()

if __name__ == "__main__":
    trainer = FakeNewsTrainer("data/news_cleaned_2018_02_13_cleaned_20pct.csv")
    trainer.run()