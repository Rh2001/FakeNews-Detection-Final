import pandas as pd
from collections import Counter

def count_labels(csv_path, label_column="type", chunksize=50000):
    label_counts = Counter()
    total_rows = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        # Drop missing/empty labels just in case
        if label_column not in chunk.columns:
            raise ValueError(f"Column '{label_column}' not found in dataset.")

        labels = chunk[label_column].dropna().astype(str).str.strip()
        labels = labels[labels != ""]

        label_counts.update(labels)
        total_rows += len(labels)

    print("\nLabel Distribution (Cleaned Data):")
    print("-" * 40)

    for label, count in label_counts.items():
        percentage = (count / total_rows) * 100 if total_rows else 0
        print(f"{label}: {count} ({percentage:.2f}%)")

    print("-" * 40)
    print(f"Total valid rows: {total_rows}")


if __name__ == "__main__":
    cleaned_file = "data/news_cleaned_2018_02_13_cleaned_20pct.csv"
    count_labels(cleaned_file)