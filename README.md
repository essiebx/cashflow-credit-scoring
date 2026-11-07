# cashflow-credit-scoring

## Alternative Data Feature Engineering & Credit Scoring Prototype

A prototype for building and evaluating machine-learning feature matrices from raw transactional logs to support small-business credit scoring.

### Target challenge

Explore whether alternative transaction data can help predict 30-day default risk (`default_flag = 1`).

### Tech stack

- Python: scikit-learn, XGBoost, and imbalanced-learn
- SQL
- Jupyter Notebook or Databricks

## Project structure

The repository currently contains only this README. The following is a suggested structure for organizing the prototype as it is developed; these files and directories are planned, not yet present.

```text
cashflow-credit-scoring/
├── README.md
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignore environments, caches, and local data
├── data/
│   ├── README.md             # Data dictionary, provenance, and access notes
│   ├── raw/                  # Local-only source data; do not commit sensitive data
│   └── processed/            # Generated feature tables; do not commit sensitive data
├── notebooks/
│   └── credit_scoring.ipynb  # End-to-end exploration, modeling, and findings
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Data validation and cleaning
│   ├── features.py           # Transaction-level feature construction
│   ├── train.py              # Training and validation workflow
│   └── evaluation.py         # Metrics, plots, and feature importance
├── sql/
│   └── transaction_aggregates.sql  # Optional SQL-based aggregation
├── reports/
│   └── model_findings.md     # One-page findings and limitations
└── tests/
    ├── test_preprocessing.py
    └── test_features.py
```

## Project workflow

### 1. Feature extraction

Transform raw transaction logs into predictive credit features, including:

- **Monthly cash-flow volatility** — variation in monthly net cash flow.
- **Average transaction value** — average value of transactions over the selected observation period.
- **Mobile-money inflow/outflow ratio** — the ratio of incoming to outgoing mobile-money transactions or value.
- **Repayment promptness score** — a measure of how promptly prior repayments were made.

Document the definitions, aggregation windows, missing-data handling, and any assumptions for each feature.

### 2. Baseline scorecard model

Train a binary classification model to estimate 30-day default risk. Start with Logistic Regression as an interpretable baseline and optionally compare it with XGBoost.

Address class imbalance using class weighting or a resampling method such as SMOTE. Apply resampling only to the training data to avoid leakage into validation or test sets.

### 3. Evaluation and feature importance

Evaluate the model on held-out data and report:

- ROC-AUC
- Precision-recall curve and relevant precision/recall metrics
- Gini coefficient
- Key feature-importance or model-coefficient findings

Describe the validation approach, class distribution, and limitations alongside the metrics. Highlight the transaction-based risk drivers that appear most informative, without treating model associations as causal conclusions.

## Deliverable

A Databricks or Jupyter notebook demonstrating:

1. Data preprocessing and quality checks
2. Feature engineering logic
3. Train/validation/test methodology and imbalance handling
4. Model training and validation metrics
5. A one-page summary of findings, key risk drivers, and limitations

## Data and responsible use

This repository describes a prototype workflow. Do not commit personally identifiable, confidential, or otherwise sensitive financial transaction data. Credit-scoring models can materially affect people and businesses; assess data quality, privacy, fairness, explainability, and applicable legal requirements before any real-world use.
