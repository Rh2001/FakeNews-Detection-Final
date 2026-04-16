import pandas as pd
import spacy
import os
from nltk.stem import PorterStemmer
from collections import Counter


class LIARPreprocessorFixed:
    def __init__(self):
        print("Loading spaCy tokenizer (aligned with FakeNews pipeline)...")

        self.nlp = spacy.blank("en")
        self.stop_words = set(self.nlp.Defaults.stop_words)
        self.stemmer = PorterStemmer()

        self.vocab_before = Counter()
        self.vocab_after_stopwords = Counter()
        self.vocab_after_stemming = Counter()

    # Label mapping similar to FakeNewsCorpus, but adapted for LIAR's labels, binary mapping
    @staticmethod
    def map_label(label):
        if pd.isna(label):
            return None

        label = str(label).lower().strip()
        label = label.replace(".", "").replace("-", " ").replace("_", " ").strip()

        label_map = {
            "false": 0,
            "pants fire": 0,
            "barely true": 0,

            "half true": 1,
            "mostly true": 1,
            "true": 1,
            "reliable": 1
        }

        return label_map.get(label, None)

    # Text cleaning aligned with the FakeNewsCorpus pipeline
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        return " ".join(text.lower().strip().split())

    # Process file
    def process_tsv(self, input_path):
        print(f"\nLoading: {input_path}")

        df = pd.read_csv(
            input_path,
            sep="\t",
            header=None,
            on_bad_lines="skip"
        )

        # LIAR format: label + statement
        df = df[[1, 2]].copy()
        df.columns = ["label", "statement"]

        df["statement"] = df["statement"].apply(self.clean_text)
        df["type"] = df["label"].apply(self.map_label)

        print("\nLabel distribution BEFORE drop:")
        print(df["type"].value_counts(dropna=False))

        df = df.dropna(subset=["type"])
        df = df[df["statement"].str.len() > 5]

        print("\nLabel distribution AFTER drop:")
        print(df["type"].value_counts())

        if len(df) == 0:
            raise ValueError("Dataset empty after preprocessing")

        texts = df["statement"].astype(str).tolist()
        processed_texts = []

        # Pipeline
        for doc in self.nlp.pipe(texts, batch_size=1000):

            before_tokens = []
            final_tokens = []

            for token in doc:
                t = token.text

                if not t.isalpha():
                    continue

                before_tokens.append(t)

                if t in self.stop_words:
                    continue

                stemmed = self.stemmer.stem(t)
                final_tokens.append(stemmed)

            self.vocab_before.update(before_tokens)
            self.vocab_after_stopwords.update(final_tokens)
            self.vocab_after_stemming.update(final_tokens)

            processed_texts.append(" ".join(final_tokens))

        df["content"] = processed_texts

        return df[["content", "type"]]

    # Save
    def run(self, input_files):
        os.makedirs("data", exist_ok=True)

        for path in input_files:
            name = os.path.basename(path).replace(".tsv", "")
            out_path = f"data/{name}_LIAR_cleaned.csv"

            df_clean = self.process_tsv(path)

            print("\nFinal size:", len(df_clean))
            print(df_clean["type"].value_counts())

            df_clean.to_csv(out_path, index=False)
            print(f"Saved → {out_path}")


if __name__ == "__main__":
    processor = LIARPreprocessorFixed()

    input_files = [
        "data/train.tsv",
        "data/valid.tsv",
        "data/test.tsv"
    ]

    processor.run(input_files)