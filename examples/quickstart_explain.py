# -*- coding: utf-8 -*-
"""
quickstart_explain.py
=====================
Minimal end-to-end example for the RAGImputer.explain() feature.

Steps
-----
1. Load the Pima Indians Diabetes dataset.
2. Introduce artificial missing values (MCAR) on the training split.
3. Fit RAGImputer on complete rows of the training set.
4. Impute the test set with transform().
5. Call explain() to get LLM-generated reasoning for each imputed row.

Requirements
------------
Set the relevant API key in a .env file at the project root, e.g.:
    API_KEY_GEMINI=<your-key>
    API_KEY_OPEN_ROUTER=<your-key>
    API_KEY_GPT=<your-key>
    API_KEY_CLAUDE=<your-key>

Run
---
    python examples/quickstart_explain.py
"""

import sys
sys.path.append("./")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from algorithms.rag_imputer import RAGImputer

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
DATA_PATH = "./data/pima-indians-diabetes/pima_diabetes.csv"
DATASET_NAME = "Pima Indians Diabetes"
MISSING_RATE = 0.15
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)

# Keep only numeric features; drop target if present
target_col = "Outcome" if "Outcome" in df.columns else None
X = df.drop(columns=[target_col]) if target_col else df.copy()
X = X.select_dtypes(include="number").astype(float)

X_train, X_test = train_test_split(X, test_size=0.1, random_state=RANDOM_STATE)

# ---------------------------------------------------------------------------
# 2. Introduce artificial missing values in the test set (MCAR)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)
X_test_missing = X_test.copy()

miss_mask = rng.random(X_test_missing.shape) < MISSING_RATE
# Ensure every masked row keeps at least one observed feature
for i, row_mask in enumerate(miss_mask):
    if row_mask.all():
        keep = rng.integers(0, X_test_missing.shape[1])
        row_mask[keep] = False
    X_test_missing.iloc[i, row_mask] = np.nan

n_missing_rows = X_test_missing.isna().any(axis=1).sum()
print(f"Test rows with missing values: {n_missing_rows}/{len(X_test_missing)}")

# ---------------------------------------------------------------------------
# 3. Fit
# ---------------------------------------------------------------------------
imputer = RAGImputer(
    n_neighbors=5,
    feature_weighting="correlation",
    llm_model_name="gemini-2.0-flash",   # change to your preferred model
    llm_api="gemini",                     # "gemini" | "open_router" | "gpt" | "claude"
    dataset_name=DATASET_NAME,
    llm_batch_size=1,
)

imputer.fit(X_train)

# ---------------------------------------------------------------------------
# 4. Impute
# ---------------------------------------------------------------------------
X_imputed = imputer.transform(X_test_missing)
X_imputed_df = pd.DataFrame(X_imputed, columns=X_test.columns, index=X_test.index)

# ---------------------------------------------------------------------------
# 5. Explain
# ---------------------------------------------------------------------------
explanations = imputer.explain(X_test_missing, X_imputed_df)

print("\n" + "=" * 70)
print("IMPUTATION EXPLANATIONS")
print("=" * 70)

missing_row_positions = X_test_missing.index[X_test_missing.isna().any(axis=1)].tolist()

for pos, (original_idx, explanation) in enumerate(
    zip(missing_row_positions, explanations)
):
    missing_cols = X_test_missing.columns[X_test_missing.loc[original_idx].isna()].tolist()
    imputed_vals = {
        col: round(X_imputed_df.loc[original_idx, col], 4) for col in missing_cols
    }

    print(f"\n[Row {original_idx}]")
    print(f"  Missing columns : {missing_cols}")
    print(f"  Imputed values  : {imputed_vals}")
    print(f"  LLM Explanation :\n  {explanation.strip()}")
    print("-" * 70)
