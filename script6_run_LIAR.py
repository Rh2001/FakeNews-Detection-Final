import os
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Metrics printing helper
def print_metrics(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="binary", zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, average="binary", zero_division=0))
    print("F1:", f1_score(y_true, y_pred, average="binary", zero_division=0))


def debug_preds(name, pred):
    unique, counts = np.unique(pred, return_counts=True)
    print(f"{name} predictions:", dict(zip(unique, counts)))


def save_cm(folder, name, y_true, y_pred):
    os.makedirs(folder, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    # Match second script style
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake", "Real"]
    )

    fig, ax = plt.subplots(figsize=(8, 6))  # bigger figure like second script
    disp.plot(ax=ax, values_format="d")

    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    path = os.path.join(folder, f"{name}.png")
    plt.savefig(path, dpi=300)  # higher quality
    plt.show()  # <-- this is key difference (also display it)

    print(f"Saved CM → {path}")


# BERT Model Wrapper
class BERTModel:
    def __init__(self, path="data/distilbert_fake_news_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts, batch_size=16):
        preds = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            enc = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits
                batch_preds = torch.argmax(logits, dim=1)

            preds.extend(batch_preds.cpu().numpy())

        return np.array(preds)


# Main evaluation function
def main():

    print("\n==============================")
    print("LIAR CROSS-DOMAIN EVAL (FIXED)")
    print("==============================")

    df = pd.read_csv("data/test_LIAR_cleaned.csv")

    print("\nRaw size:", len(df))

    df = df.dropna(subset=["content", "type"]).copy()

    # enforce consistency
    df["content"] = df["content"].astype(str).str.lower().str.strip()

    df["binary_label"] = df["type"].astype(int)

    X = df["content"].values
    y_true = df["binary_label"].values

    print("\nClass distribution:", np.unique(y_true, return_counts=True))

    assert len(X) == len(y_true), "Feature/label mismatch"
    assert not np.any(pd.isna(y_true)), "NaN labels found"

    # Load models
    content_lr = joblib.load("content_baseline_models/logistic_regression_model.joblib")
    content_nb = joblib.load("content_baseline_models/naive_bayes_model.joblib")

    meta_lr = joblib.load("meta_baseline_models/logistic_regression_model.joblib")
    meta_nb = joblib.load("meta_baseline_models/naive_bayes_model.joblib")

    bert = BERTModel()

    # Predictions
    content_lr_pred = content_lr.predict(X)
    content_nb_pred = content_nb.predict(X)

    meta_lr_pred = meta_lr.predict(X)
    meta_nb_pred = meta_nb.predict(X)

    bert_pred = bert.predict(X.tolist())

    # Debugging prediction distributions
    print("\n===== PREDICTION DISTRIBUTIONS =====")
    debug_preds("CONTENT LR", content_lr_pred)
    debug_preds("CONTENT NB", content_nb_pred)
    debug_preds("META LR", meta_lr_pred)
    debug_preds("META NB", meta_nb_pred)
    debug_preds("BERT", bert_pred)

    # Metrics
    print_metrics("CONTENT LR", y_true, content_lr_pred)
    print_metrics("CONTENT NB", y_true, content_nb_pred)
    print_metrics("META LR", y_true, meta_lr_pred)
    print_metrics("META NB", y_true, meta_nb_pred)
    print_metrics("BERT", y_true, bert_pred)

    # Confusion matrix(this one is saved in cm to separate it from the other ones )
    save_cm("Confusion_Matrices/LIAR/Content", "content_lr", y_true, content_lr_pred)
    save_cm("Confusion_Matrices/LIAR/Content", "content_nb", y_true, content_nb_pred)

    save_cm("Confusion_Matrices/LIAR/Meta", "meta_lr", y_true, meta_lr_pred)
    save_cm("Confusion_Matrices/LIAR/Meta", "meta_nb", y_true, meta_nb_pred)

    save_cm("Confusion_Matrices/LIAR/BERT", "bert", y_true, bert_pred)


if __name__ == "__main__":
    main()