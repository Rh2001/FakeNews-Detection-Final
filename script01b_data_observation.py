import pandas as pd
import re

# Configuration
INPUT_PATH = "data/news_cleaned_2018_02_13_cleaned_20pct.csv"

TEXT_COLUMNS = ["domain" ,"content", "title", "authors", "keywords", "source"]
LABEL_COLUMN = "type"

URL_PATTERN = re.compile(
    r"""(
        https?://\S+              |   # full URLs
        www\.\S+                 |   # www links
        \b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}/\S* |  # domain with path
        \b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}\b      # bare domains
    )""",
    re.VERBOSE
)
HTML_PATTERN = re.compile(r"<[^>]+>")

# Load data in chunks
def load_data(path, chunksize=50_000):
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

df = load_data(INPUT_PATH)

print("\n================ DATASET OVERVIEW ================\n")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(df.columns)

# Missing labels section
missing_labels = (
    df[LABEL_COLUMN].isna().sum()
    + df[LABEL_COLUMN].astype(str).str.strip().eq("").sum()
)

print("\n================ LABEL QUALITY ================\n")
print(f"Missing or empty labels: {missing_labels}")
print(f"Percentage missing labels: {missing_labels / len(df):.2%}")

# URL detection
def count_urls(series):
    return series.fillna("").astype(str).str.count(URL_PATTERN).sum()

print("\n================ URL STATISTICS ================\n")

for col in TEXT_COLUMNS:
    if col in df.columns:
        url_count = count_urls(df[col])
        print(f"{col}: {url_count} URLs")

# HTML section
def count_html(series):
    return series.fillna("").astype(str).str.count(HTML_PATTERN).sum()

print("\n================ HTML TAG STATISTICS ================\n")

for col in TEXT_COLUMNS:
    if col in df.columns:
        html_count = count_html(df[col])
        print(f"{col}: {html_count} HTML tag occurrences")

# Sparity section
print("\n================ TEXT SPARSITY ================\n")

empty_rates = {}

for col in TEXT_COLUMNS:
    if col in df.columns:

        empty_mask = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        empty_count = empty_mask.sum()
        empty_rate = empty_count / len(df)

        empty_rates[col] = empty_rate

        print(f"{col}: {empty_count} empty values ({empty_rate:.2%})")

# Most sparse columns
if empty_rates:
    worst_col = max(empty_rates, key=empty_rates.get)

    print("\n================ MOST SPARSE COLUMN ================\n")
    print(f"Most empty column: {worst_col}")
    print(f"Empty rate: {empty_rates[worst_col]:.2%}")

# Label distribution
print("\n================ LABEL DISTRIBUTION ================\n")
print(df[LABEL_COLUMN].value_counts(dropna=False).head(20))