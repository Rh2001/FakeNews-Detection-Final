"""
Updated Preprocessing script for fake news dataset.
Processes large CSV in chunks, tokenizes in parallel with spaCy, removes stopwords, keeps HTML and UML for analysis(they're commented)
applies stemming, and saves processed chunks incrementally.
"""

import pandas as pd
import csv
import spacy
import spacy.cli
from nltk.stem import PorterStemmer
from collections import Counter
import os
import time

class FakeNewsPreprocessor:
    def __init__(self, n_process=4):
        print("Loading spaCy English model...")
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

        self.stop_words = self.nlp.Defaults.stop_words
        self.stemmer = PorterStemmer()

        # Columns to process
        self.text_columns = ["content", "title", "authors", "keywords", "source"]
        self.label_column = "type"

        # Vocabulary counters
        self.vocab_before = Counter()
        self.vocab_after_stopwords = Counter()
        self.vocab_after_stemming = Counter()

        # Number of parallel processes
        self.n_process = n_process

    def clean_text_series(self, series):
        """Fill NaN, lowercase text."""
        s = series.fillna("").str.lower()
        # Remove URLs and HTML if needed
        # s = s.str.replace(r"http\S+|www\S+", "", regex=True)
        # s = s.str.replace(r"<.*?>", "", regex=True)
        return s

    def process_chunk(self, chunk):
        """Clean, tokenize, remove stopwords, stem, update vocab, return processed DataFrame."""

        # Clean text columns
        for col in self.text_columns:
            if col in chunk.columns:
                chunk[col] = self.clean_text_series(chunk[col])

        # Filter invalid or empty content and labels
        if "content" in chunk.columns:
            chunk = chunk[chunk["content"].str.strip() != ""]
        if self.label_column in chunk.columns:
            chunk = chunk[chunk[self.label_column].astype(str).str.strip() != ""]
        chunk = chunk.drop_duplicates(subset=["content", "title"])

        # Tokenize, remove stopwords, stem, update vocab
        for col in self.text_columns:
            if col in chunk.columns:
                processed_texts = []

                # spaCy parallel tokenization
                for doc in self.nlp.pipe(chunk[col], batch_size=500, n_process=self.n_process):
                    tokens = [token.text for token in doc if token.is_alpha]

                    # Update vocab
                    self.vocab_before.update(tokens)

                    # Remove stopwords
                    tokens = [w for w in tokens if w not in self.stop_words]
                    self.vocab_after_stopwords.update(tokens)

                    # Apply stemming
                    tokens = [self.stemmer.stem(w) for w in tokens]
                    self.vocab_after_stemming.update(tokens)

                    processed_texts.append(" ".join(tokens))

                chunk[col] = processed_texts

        return chunk

    def load_and_process(self, input_csv, output_csv, chunksize=30_000, sample_frac=0.1):
        """Process CSV in memory-efficient batches, save to disk incrementally."""

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

        # Dynamically detect available columns
        all_cols = pd.read_csv(input_csv, nrows=1).columns
        used_cols = [col for col in self.text_columns + [self.label_column] if col in all_cols]

        # Save and print first row of the original dataset (all columns, prior to removal)
        first_row_full = pd.read_csv(input_csv, nrows=1)
        print("First row of dataset (all columns, before any removal):")
        print(first_row_full)
        first_row_full.to_csv(first_row_full_path, index=False)
        print(f"Saved first row of entire dataset to: {first_row_full_path}")

        for chunk in pd.read_csv(
            input_csv,
            usecols=used_cols,
            chunksize=chunksize,
            engine="python",
            on_bad_lines="skip"
        ):
            batch_num += 1

            # Save first row of chunk for inspection
            first_row = chunk.head(1)
            first_row.to_csv(
                first_rows_path,
                mode="a",
                index=False,
                header=not os.path.exists(first_rows_path),
            )

            print(f"\nProcessing chunk {batch_num}...")

            # Sample fraction to reduce memory usage
            chunk = chunk.sample(frac=sample_frac, random_state=42)
            chunk = self.process_chunk(chunk)

            # Append processed chunk to CSV
            write_header = not os.path.exists(output_csv)
            chunk.to_csv(output_csv, mode="a", index=False, header=write_header)

            print(f"Chunk {batch_num} processed and saved.")

        elapsed = time.time() - start_time
        print(f"\nAll batches processed in {elapsed:.2f} seconds.")
        print(f"Processed data saved to: {output_csv}")

        self.report_vocab_statistics()

    def report_vocab_statistics(self):
        """Print vocabulary sizes and reduction rates(I'll have to rerun this later on with a smaller sample of the dataset)."""
        vocab_before_size = len(self.vocab_before)
        vocab_after_stop_size = len(self.vocab_after_stopwords)
        vocab_after_stem_size = len(self.vocab_after_stemming)

        stopword_reduction = 1 - vocab_after_stop_size / vocab_before_size if vocab_before_size else 0
        stemming_reduction = 1 - vocab_after_stem_size / vocab_after_stop_size if vocab_after_stop_size else 0

        print("\n Vocabulary Statistics:")
        print(f"Vocabulary size before stopwords: {vocab_before_size}")
        print(f"Vocabulary size after stopwords removal: {vocab_after_stop_size}")
        print(f"Reduction rate after stopwords removal: {stopword_reduction:.2%}")
        print(f"Vocabulary size after stemming: {vocab_after_stem_size}")
        print(f"Reduction rate after stemming: {stemming_reduction:.2%}")


# Required on Windows, not in Linux/Mac, to avoid multiprocessing issues
if __name__ == "__main__":
    # Initialize processor with all available CPU processes 
    processor = FakeNewsPreprocessor(n_process=os.cpu_count())
    
    input_path = "data/news_cleaned_2018_02_13.csv"
    output_path = "data/news_cleaned_2018_02_13_cleaned_10pct.csv"

    processor.load_and_process(
        input_csv=input_path,
        output_csv=output_path,
        chunksize=30_000,
        sample_frac=0.1
    )