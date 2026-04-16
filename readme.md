# FakeNews-Detection

End-to-end fake news classification pipeline using:

- Traditional ML baselines (TF-IDF + Logistic Regression / Naive Bayes)
- DistilBERT fine-tuning
- Cross-domain evaluation on LIAR

## Project Goal

Train models on a cleaned FakeNewsCorpus-style dataset, then evaluate how well they generalize to LIAR claims (cross-domain transfer).

## Repository Structure

- [script01_preprocessing_fakenewscorpus.py](script01_preprocessing_fakenewscorpus.py): Main preprocessing for large fake news CSV (20% chunk sampling, cleaning, stopword removal, stemming).
- [script02_simplemodels.py](script02_simplemodels.py): Baseline models (content-only and content+metadata), metrics, confusion matrices, model export.
- [script03_advancedmodel_destilbert.py](script03_advancedmodel_destilbert.py): DistilBERT training with weighted loss and early stopping.
- [script04_evaluate_destilbert.py](script04_evaluate_destilbert.py): Standalone DistilBERT evaluation and confusion matrix display.
- [script05b_download_LIAR.py](script05b_download_LIAR.py): Download and extract raw LIAR dataset.
- [script05_preprocessing_LIAR.py](script05_preprocessing_LIAR.py): Clean LIAR TSV files and map labels to binary.
- [script6_run_LIAR.py](script6_run_LIAR.py): Cross-domain evaluation on LIAR using saved baseline models and saved DistilBERT model.
- [setup.py](setup.py): Downloads required NLTK resources.
- [requirements.txt](requirements.txt): Base dependencies list.

## Technical Decisions

### Text Preprocessing Pipeline

- Uses spaCy tokenizer in fast mode with spacy.blank("en") for speed on large corpora.
- Lowercases and normalizes text before token filtering.
- Removes non-alphabetic tokens with str.isalpha().
- Removes stopwords using spaCy default English stopword list.
- Applies Porter stemming for normalization.
- Processes data in chunks to reduce memory pressure on large CSV files.

### Label Normalization Strategy

FakeNewsCorpus labels are mapped to binary:
- 0: fake, rumor, conspiracy, junksci
- 1: reliable, political

LIAR labels are mapped to binary:
- 0: false, pants fire, barely true
- 1: half true, mostly true, true, reliable

Unknown labels are dropped to avoid noisy supervision.

### Baseline Model Design

- Uses TF-IDF with two classic baselines:
  - Logistic Regression
  - Multinomial Naive Bayes
- Trains two variants:
  - Content-only text
  - Content + metadata concatenation (domain, title, authors, keywords, source)
- Saves trained baseline models for later reuse in cross-domain testing.

### DistilBERT Design Choices

- Uses distilbert-base-uncased with a binary classification head.
- Tokenization is batched and truncated to max length 256 to control memory and compute.
- Uses class-weighted cross-entropy via a custom Trainer to handle class imbalance.
- Uses early stopping and best-model selection by validation F1.
- Saves tokenizer and model under the data/distilbert_fake_news_model directory.

### Cross-Domain Evaluation Design

- Evaluates all saved models on LIAR test set from [script6_run_LIAR.py](script6_run_LIAR.py).
- Reports Accuracy, Precision, Recall, and F1.
- Prints prediction distribution per model for debugging collapse and bias.
- Saves confusion matrices in cm/Content, cm/Meta, and cm/BERT.

## Environment Setup (Windows PowerShell)

1. Create and activate virtual environment

    python -m venv .venv  
    .\.venv\Scripts\Activate.ps1

2. Install dependencies from requirements

    pip install -r requirements.txt

3. Install missing runtime packages used by scripts

    pip install torch transformers matplotlib joblib requests

4. Download NLTK resources

    python setup.py

## Full Pipeline (Recommended Run Order)

1. Preprocess FakeNews corpus

    python script01_preprocessing_fakenewscorpus.py

Expected output:
- data/news_cleaned_2018_02_13_cleaned_20pct.csv

2. Train baseline models (content and metadata variants)

    python script02_simplemodels.py

Expected outputs:
- content_baseline_models/logistic_regression_model.joblib
- content_baseline_models/naive_bayes_model.joblib
- meta_baseline_models/logistic_regression_model.joblib
- meta_baseline_models/naive_bayes_model.joblib
- Confusion matrices under the Confusion_Matrices directory

3. Train DistilBERT

    python script03_advancedmodel_destilbert.py

Expected outputs:
- data/distilbert_fake_news_model/config.json
- data/distilbert_fake_news_model/model.safetensors
- data/distilbert_fake_news_model/tokenizer.json
- Checkpoints under results

4. Evaluate DistilBERT on FakeNews test split (optional)

    python script04_evaluate_destilbert.py

5. Download LIAR raw files (if not already present)

    python script05b_download_LIAR.py

6. Preprocess LIAR data

    python script05_preprocessing_LIAR.py

Expected outputs:
- data/train_LIAR_cleaned.csv
- data/valid_LIAR_cleaned.csv
- data/test_LIAR_cleaned.csv

7. Run cross-domain evaluation on LIAR

    python script6_run_LIAR.py

Expected outputs:
- Printed metrics for CONTENT LR, CONTENT NB, META LR, META NB, and BERT
- Confusion matrices:
  - cm/Content/content_lr.png
  - cm/Content/content_nb.png
  - cm/Meta/meta_lr.png
  - cm/Meta/meta_nb.png
  - cm/BERT/bert.png

## Run Only the LIAR Cross-Domain Evaluation

### Prerequisites

- data/test_LIAR_cleaned.csv exists (from [script05_preprocessing_LIAR.py](script05_preprocessing_LIAR.py))
- Saved baseline models exist in content_baseline_models and meta_baseline_models
- Saved DistilBERT model exists in data/distilbert_fake_news_model

### Command

    python script6_run_LIAR.py

### What It Does

- Loads LIAR cleaned test set
- Loads 5 trained models (2 content baselines, 2 metadata baselines, 1 DistilBERT)
- Generates predictions
- Prints metric summary
- Saves confusion matrix images

## Known Notes

- [script6_run_LIAR.py](script6_run_LIAR.py) uses only the content column for all baseline predictions; metadata-trained models were trained on concatenated features, so cross-domain performance can degrade significantly due to feature mismatch between the two datasets.
- [.gitignore](.gitignore) currently ignores all CSV files,tsv files and dataset, so generated cleaned datasets are not tracked by git.