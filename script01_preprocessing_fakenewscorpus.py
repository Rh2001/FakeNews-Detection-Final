"""
Faster Preprocessing script for fake news dataset.
Changes:
- spaCy tokenizer only (spacy.blank)
- NO shuffling.
- 20% sampling per chunk.
- domain handled separately (categorical feature)
- Faster token checks (str.isalpha)
- I changed back to python engine because it crashed often with C on large chunks.
- 20% of the whole dataset for my system took around 2-3 hours
"""

import pandas as pd
import csv
import spacy
from nltk.stem import PorterStemmer
from collections import Counter
import os
import time


class FakeNewsPreprocessor:
    def __init__(self, n_process=4):
        print("Loading spaCy tokenizer (fast mode)...")

        self.nlp = spacy.blank("en")
        self.stop_words = set(self.nlp.Defaults.stop_words)

        self.stemmer = PorterStemmer()

        self.text_columns = ["content", "title", "authors", "keywords", "source"]
        self.categorical_columns = ["domain"]

        self.label_column = "type"

        self.vocab_before = Counter()
        self.vocab_after_stopwords = Counter()
        self.vocab_after_stemming = Counter()

        self.n_process = n_process

    def clean_text_series(self, series):
        s = series.fillna("")
        s = s.astype(str)
        return s.str.lower()

    def clean_domain(self, series):
        return (
            series.fillna("")
            .astype(str)
            .str.lower()
            .str.replace("www.", "", regex=False)
            .str.strip()
        )

    def process_chunk(self, chunk):

        active_cols = []
        for col in self.text_columns:
            if col in chunk.columns:
                chunk[col] = self.clean_text_series(chunk[col])
                active_cols.append(col)

        if "domain" in chunk.columns:
            chunk["domain"] = self.clean_domain(chunk["domain"])

        if "content" in chunk.columns:
            chunk = chunk[chunk["content"].str.strip() != ""]
        if self.label_column in chunk.columns:
            chunk = chunk[chunk[self.label_column].astype(str).str.strip() != ""]

        chunk = chunk.drop_duplicates(subset=["content", "title"])

        texts = []
        col_lengths = []

        for col in active_cols:
            col_data = chunk[col].tolist()
            texts.extend(col_data)
            col_lengths.append(len(col_data))

        stop_words = self.stop_words
        stem = self.stemmer.stem

        vocab_before_batch = []
        vocab_after_stop_batch = []
        vocab_after_stem_batch = []

        processed_texts = []

        for doc in self.nlp.pipe(texts, batch_size=5000):

            before = []
            after = []
            final = []

            for token in doc:
                text = token.text

                if not text.isalpha():
                    continue

                before.append(text)

                if text in stop_words:
                    continue

                after.append(text)

                final.append(stem(text))

            vocab_before_batch.extend(before)
            vocab_after_stop_batch.extend(after)
            vocab_after_stem_batch.extend(final)

            processed_texts.append(" ".join(final))

        self.vocab_before.update(vocab_before_batch)
        self.vocab_after_stopwords.update(vocab_after_stop_batch)
        self.vocab_after_stemming.update(vocab_after_stem_batch)

        idx = 0
        for col, length in zip(active_cols, col_lengths):
            chunk[col] = processed_texts[idx:idx + length]
            idx += length

        return chunk

    def load_and_process(self, input_csv, output_csv, chunksize=30_000, sample_frac=0.2):

        csv.field_size_limit(10_000_000)

        if os.path.exists(output_csv):
            os.remove(output_csv)

        first_rows_path = "data/chunk_first_rows.csv"
        if os.path.exists(first_rows_path):
            os.remove(first_rows_path)

        first_row_full_path = "data/first_row_full.csv"
        if os.path.exists(first_row_full_path):
            os.remove(first_row_full_path)

        start_time = time.time()
        batch_num = 0

        all_cols = pd.read_csv(input_csv, nrows=1).columns

        used_cols = list(set(self.text_columns + self.categorical_columns + [self.label_column]))
        used_cols = [col for col in used_cols if col in all_cols]

        first_row_full = pd.read_csv(input_csv, nrows=1)
        print("First row of dataset (all columns):")
        print(first_row_full)
        first_row_full.to_csv(first_row_full_path, index=False)

        for chunk in pd.read_csv(
            input_csv,
            usecols=used_cols,
            chunksize=chunksize,
            engine="python",
            on_bad_lines="skip"
        ):
            batch_num += 1

            first_row = chunk.head(1)
            first_row.to_csv(
                first_rows_path,
                mode="a",
                index=False,
                header=not os.path.exists(first_rows_path),
            )

            print(f"\nProcessing chunk {batch_num}...")

            if sample_frac < 1.0:
                chunk = chunk.sample(frac=sample_frac, random_state=42)

            chunk = self.process_chunk(chunk)

            write_header = not os.path.exists(output_csv)
            chunk.to_csv(output_csv, mode="a", index=False, header=write_header)

            print(f"Chunk {batch_num} processed and saved.")

        elapsed = time.time() - start_time
        print(f"\nAll batches processed in {elapsed:.2f} seconds.")
        print(f"Processed data saved to: {output_csv}")

        self.report_vocab_statistics()

    def report_vocab_statistics(self):
        # Unique vocab sizes (your original metrics)
        vocab_before_size = len(self.vocab_before)
        vocab_after_stop_size = len(self.vocab_after_stopwords)
        vocab_after_stem_size = len(self.vocab_after_stemming)

        # ✅ NEW: total token counts (correct metric)
        total_before = sum(self.vocab_before.values())
        total_after_stop = sum(self.vocab_after_stopwords.values())
        total_after_stem = sum(self.vocab_after_stemming.values())

        # Reductions
        stopword_reduction_vocab = (
            1 - vocab_after_stop_size / vocab_before_size
            if vocab_before_size else 0
        )

        stemming_reduction_vocab = (
            1 - vocab_after_stem_size / vocab_after_stop_size
            if vocab_after_stop_size else 0
        )

        stopword_reduction_tokens = (
            1 - total_after_stop / total_before
            if total_before else 0
        )

        stemming_reduction_tokens = (
            1 - total_after_stem / total_after_stop
            if total_after_stop else 0
        )

        print("\nVocabulary Statistics:")

        print("\n--- UNIQUE VOCAB SIZE ---")
        print(f"Before stopwords: {vocab_before_size}")
        print(f"After stopwords: {vocab_after_stop_size}")
        print(f"Reduction: {stopword_reduction_vocab:.2%}")
        print(f"After stemming: {vocab_after_stem_size}")
        print(f"Reduction: {stemming_reduction_vocab:.2%}")

        print("\n--- TOKEN COUNTS (CORRECT METRIC) ---")
        print(f"Before stopwords: {total_before}")
        print(f"After stopwords: {total_after_stop}")
        print(f"Reduction: {stopword_reduction_tokens:.2%}")
        print(f"After stemming: {total_after_stem}")
        print(f"Reduction: {stemming_reduction_tokens:.2%}")


if __name__ == "__main__":
    processor = FakeNewsPreprocessor(n_process=os.cpu_count())

    input_path = "data/news_cleaned_2018_02_13.csv"
    output_path = "data/news_cleaned_2018_02_13_cleaned_20pct.csv"

    processor.load_and_process(
        input_csv=input_path,
        output_csv=output_path,
        chunksize=30_000,
        sample_frac=0.2
    )