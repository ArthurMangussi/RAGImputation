"""Smoke test for enhanced RAGImputer (embeddings + FAISS)."""
import sys
sys.path.insert(0, "./")

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# ------------------------------------------------------------------
# 1. Load iris complete reference
# ------------------------------------------------------------------
data = load_iris()
X_full = pd.DataFrame(data.data, columns=data.feature_names)

# ------------------------------------------------------------------
# 2. Introduce missings (~10 % of cells)
# ------------------------------------------------------------------
rng = np.random.default_rng(42)
X_missing = X_full.copy()
mask = rng.random(X_full.shape) < 0.10
X_missing[mask] = np.nan
print(f"NaN cells introduced: {X_missing.isna().sum().sum()}")

# ------------------------------------------------------------------
# 3. Test aggregation mode with FAISS flat index
# ------------------------------------------------------------------
from algorithms.rag_imputer import RAGImputer

print("\n[1] Aggregation mode – FAISS flat index")
imputer = RAGImputer(n_neighbors=5, mode="aggregation", faiss_index_type="flat")
imputer.fit(X_full)

print(f"    Context store size : {imputer.context_store_.shape}")
print(f"    Embedding dim      : {imputer.faiss_index_.d}")
print(f"    FAISS total vecs   : {imputer.faiss_index_.ntotal}")

X_imputed = imputer.transform(X_missing)
df_imp = pd.DataFrame(X_imputed, columns=X_full.columns)
assert not df_imp.isna().any().any(), "Still has NaNs!"
print("    [OK] No NaNs remaining")

# ------------------------------------------------------------------
# 4. Test aggregation mode with FAISS IVF index (>1000 rows forced)
# ------------------------------------------------------------------
print("\n[2] Aggregation mode – FAISS IVF index (simulated large dataset)")
X_large = pd.DataFrame(
    np.tile(X_full.values, (10, 1)),   # 1500 rows – triggers IVF
    columns=X_full.columns
)
imputer_ivf = RAGImputer(n_neighbors=5, mode="aggregation", faiss_index_type="ivf")
imputer_ivf.fit(X_large)
X_imp_ivf = imputer_ivf.transform(X_missing)
assert not pd.DataFrame(X_imp_ivf).isna().any().any(), "IVF still has NaNs!"
print("    [OK] IVF index – no NaNs remaining")

# ------------------------------------------------------------------
# 5. Test fit on partially complete data (complete rows filtered)
# ------------------------------------------------------------------
print("\n[3] Fit on partially missing training data")
imputer2 = RAGImputer(n_neighbors=3)
imputer2.fit(X_missing)   # should use only complete rows
X_imp2 = imputer2.transform(X_missing)
assert not pd.DataFrame(X_imp2).isna().any().any(), "Partial-fit still has NaNs!"
print("    [OK] No NaNs remaining")

# ------------------------------------------------------------------
# 6. sklearn API compliance
# ------------------------------------------------------------------
print("\n[4] sklearn API checks")
from sklearn.base import clone
cloned = clone(imputer)
assert cloned.n_neighbors == imputer.n_neighbors
assert cloned.embedding_model == imputer.embedding_model
print("    [OK] clone works")

imputer.fit(X_full)
names = imputer.get_feature_names_out()
assert names == list(X_full.columns)
print("    [OK] get_feature_names_out works")

# ------------------------------------------------------------------
# 7. Entirely missing row fallback
# ------------------------------------------------------------------
print("\n[5] Entirely missing row fallback")
X_all_nan = X_missing.copy()
X_all_nan.iloc[0, :] = np.nan  # make row 0 entirely NaN
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    X_imp3 = imputer.transform(X_all_nan)
assert not pd.DataFrame(X_imp3).isna().any().any(), "All-NaN fallback still has NaNs!"
assert any("no observed features" in str(w.message).lower() for w in caught), \
    "Expected fallback warning not raised"
print("    [OK] All-NaN row filled with context mean + warning raised")

print("\n\nAll checks PASSED ✓")
