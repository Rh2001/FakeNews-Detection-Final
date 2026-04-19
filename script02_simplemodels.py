import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


class FakeNewsClassifier:
    """Fake News Classification with Baselines + Metadata + Confusion Matrix"""

    def __init__(self, data_path: str = "data/cleaned_data.csv") -> None:
        self.data_path = data_path

        # Data
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # Logistic Regression PipeLine, clf stands for classifier(logistic regression in this case)
        self.lr_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

        # Naive Bayes PipeLine, clf stands for classifier(naive bayes in this case)
        self.nb_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB()),
        ])

        self.best_model = None

        # NEW: store predictions (so we don't retrain)
        self.lr_val_pred = None
        self.lr_test_pred = None
        self.nb_val_pred = None
        self.nb_test_pred = None

        # experiment tracking
        self.experiment_name = None

    # Map labels
    @staticmethod
    def map_label(label):
        label = str(label).lower()

        # Labels that indicate fake news
        if label in ["fake", "rumor", "conspiracy", "junksci"]:
            return 0
        # Labels that indicate real news
        if label in ["reliable", "political"]:
            return 1
        return None

    # Load and split data, with option to include metadata features
    def load_and_split_data(self, use_metadata=False):

        # 🔥 THIS IS THE ONLY SOURCE OF TRUTH
        if use_metadata:
            self.experiment_name = "metadata"
            print("\nUsing CONTENT + METADATA features...")
        else:
            self.experiment_name = "content"
            print("\nUsing CONTENT ONLY...")

        df = pd.read_csv(self.data_path)

        df["binary_label"] = df["type"].apply(self.map_label)
        df = df.dropna(subset=["binary_label"]).copy()

        df["content"] = df["content"].fillna("")

        if use_metadata:
            df["domain"] = df["domain"].fillna("")
            df["title"] = df["title"].fillna("")
            df["authors"] = df["authors"].fillna("")
            df["keywords"] = df["keywords"].fillna("")
            df["source"] = df["source"].fillna("")

            df["text"] = (
                df["domain"] + " " +
                df["content"] + " " +
                df["title"] + " " +
                df["authors"] + " " +
                df["keywords"] + " " +
                df["source"]
            )
        else:
            df["text"] = df["content"]

        X = df["text"]
        y = df["binary_label"]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        self.df = df
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        print("Train:", len(self.X_train))
        print("Validation:", len(self.X_val))
        print("Test:", len(self.X_test))

    # NEW: helper to save confusion matrices with PERFECT separation
    def _save_conf_matrix(self, y_true, y_pred, filename):
        base_dir = f"Confusion_Matrices/{self.experiment_name}"
        os.makedirs(base_dir, exist_ok=True)

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # Ensure consistent order: Real (1) first, Fake (0) second
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])

        fig, ax = plt.subplots()
        disp.plot(ax=ax, values_format="d")
        ax.set_title(filename.replace("_", " "))

        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, filename))
        plt.close()

    # Baseline models: Logistic Regression and Naive Bayes
    def train_baselines(self):
        print("\n===== BASELINE MODELS =====")

        print("\nTraining Logistic Regression...")
        self.lr_pipeline.fit(self.X_train, self.y_train)

        self.lr_val_pred = self.lr_pipeline.predict(self.X_val)
        self.lr_test_pred = self.lr_pipeline.predict(self.X_test)

        print("\nLogistic Regression (Validation):")
        self.print_metrics(self.y_val, self.lr_val_pred)
        self._save_conf_matrix(self.y_val, self.lr_val_pred, "lr_validation_confusion_matrix.png")

        print("\nLogistic Regression (Test):")
        self.print_metrics(self.y_test, self.lr_test_pred)
        self._save_conf_matrix(self.y_test, self.lr_test_pred, "lr_test_confusion_matrix.png")

        print("\nTraining Naive Bayes...")
        self.nb_pipeline.fit(self.X_train, self.y_train)

        self.nb_val_pred = self.nb_pipeline.predict(self.X_val)
        self.nb_test_pred = self.nb_pipeline.predict(self.X_test)

        print("\nNaive Bayes (Validation):")
        self.print_metrics(self.y_val, self.nb_val_pred)
        self._save_conf_matrix(self.y_val, self.nb_val_pred, "nb_validation_confusion_matrix.png")

        print("\nNaive Bayes (Test):")
        self.print_metrics(self.y_test, self.nb_test_pred)
        self._save_conf_matrix(self.y_test, self.nb_test_pred, "nb_test_confusion_matrix.png")

        # SAVE MODELS (CLEAN SPLIT, NO GUESSING)
        if self.experiment_name == "content":
            save_dir = "content_baseline_models"
        else:
            save_dir = "meta_baseline_models"

        os.makedirs(save_dir, exist_ok=True)

        joblib.dump(self.lr_pipeline, os.path.join(save_dir, "logistic_regression_model.joblib"))
        joblib.dump(self.nb_pipeline, os.path.join(save_dir, "naive_bayes_model.joblib"))

        self.compare_models_on_test(self.y_test, self.lr_test_pred, self.nb_test_pred)

        self.plot_confusion_matrices_side_by_side(
            self.y_test,
            self.lr_test_pred,
            self.nb_test_pred
        )

    # Model comparison table on test set
    def compare_models_on_test(self, y_true, lr_pred, nb_pred):
        print("\n===== MODEL COMPARISON (TEST SET) =====")

        results = pd.DataFrame({
            "Model": ["Logistic Regression", "Naive Bayes"],
            "Accuracy": [
                accuracy_score(y_true, lr_pred),
                accuracy_score(y_true, nb_pred),
            ],
            "Precision": [
                precision_score(y_true, lr_pred, pos_label=0),  # Focus on fake news precision
                precision_score(y_true, nb_pred, pos_label=0),
            ],
            "Recall": [
                recall_score(y_true, lr_pred, pos_label=0),  # Focus on fake news recall
                recall_score(y_true, nb_pred, pos_label=0),
            ],
            "F1 Score": [
                f1_score(y_true, lr_pred, pos_label=0),  # Focus on fake news F1
                f1_score(y_true, nb_pred, pos_label=0),
            ],
        })

        print(results.to_string(index=False))

    # Side-by-side confusion matrix
    def plot_confusion_matrices_side_by_side(self, y_true, lr_pred, nb_pred):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        lr_cm = confusion_matrix(y_true, lr_pred, labels=[1, 0])
        nb_cm = confusion_matrix(y_true, nb_pred, labels=[1, 0])

        ConfusionMatrixDisplay(lr_cm, display_labels=["Real", "Fake"]).plot(ax=axes[0], values_format="d")
        axes[0].set_title("Logistic Regression")

        ConfusionMatrixDisplay(nb_cm, display_labels=["Real", "Fake"]).plot(ax=axes[1], values_format="d")
        axes[1].set_title("Naive Bayes")

        os.makedirs(f"Confusion_Matrices/{self.experiment_name}", exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"Confusion_Matrices/{self.experiment_name}/lr_nb_test_comparison_confusion_matrix.png"
        )
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix (Positive: Fake)"):
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])

        disp.plot()
        plt.title(title)
        plt.show()

    def evaluate(self):
        print("\n===== FINAL EVALUATION (LOGISTIC REGRESSION) =====")

        val_pred = self.lr_val_pred
        test_pred = self.lr_test_pred

        print("\nValidation:")
        self.print_metrics(self.y_val, val_pred)

        self.plot_confusion_matrix(self.y_val, val_pred, title="Validation Confusion Matrix")

        print("\nTest:")
        self.print_metrics(self.y_test, test_pred)

        self.plot_confusion_matrix(self.y_test, test_pred, title="Test Confusion Matrix")

        print("\nDetailed Report:")
        print(classification_report(
            self.y_test,
            test_pred,
            labels=[0, 1],
            target_names=["Fake", "Real"]
            ))

    @staticmethod
    def print_metrics(y_true, y_pred):
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Precision:", precision_score(y_true, y_pred, pos_label=0))
        print("Recall:", recall_score(y_true, y_pred, pos_label=0))
        print("F1 Score:", f1_score(y_true, y_pred, pos_label=0))

    def show_top_words(self, top_n=10):
        print("\n===== IMPORTANT WORDS =====")

        feature_names = self.best_model.named_steps["tfidf"].get_feature_names_out()
        coef = self.best_model.named_steps["clf"].coef_[0]

        top_fake = np.argsort(coef)[:top_n]
        top_real = np.argsort(coef)[-top_n:]

        print("\nTop FAKE words:")
        print([feature_names[i] for i in top_fake])

        print("\nTop RELIABLE words:")
        print([feature_names[i] for i in top_real])

    def run(self):
        print("\n============================")
        print("TASK 1: CONTENT ONLY")
        print("============================")

        self.load_and_split_data(use_metadata=False)
        self.train_baselines()

        if f1_score(self.y_val, self.lr_val_pred, pos_label=0) > \
            f1_score(self.y_val, self.nb_val_pred, pos_label=0):
            self.best_model = self.lr_pipeline
        else:
             self.best_model = self.nb_pipeline
        self.evaluate()
        self.show_top_words()

        print("\n============================")
        print("TASK 2: WITH METADATA")
        print("============================")

        self.load_and_split_data(use_metadata=True)
        self.train_baselines()

        if f1_score(self.y_val, self.lr_val_pred, pos_label=0) > \
            f1_score(self.y_val, self.nb_val_pred, pos_label=0):
            self.best_model = self.lr_pipeline
        else:
            self.best_model = self.nb_pipeline
        
        self.evaluate()


# Main execution
if __name__ == "__main__":
    classifier = FakeNewsClassifier("data/news_cleaned_2018_02_13_cleaned_20pct.csv")
    classifier.run()