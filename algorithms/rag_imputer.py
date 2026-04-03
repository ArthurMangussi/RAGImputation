# -*- coding: utf-8 -*-
from __future__ import annotations

# =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = "Arthur Dantas Mangussi"

"""
RAGImputer – Retrieval-Augmented Generation for Missing Data Imputation
========================================================================

Full RAG pipeline for tabular missing-data imputation:

    Encoding   – rows serialised to text → sentence-transformer embeddings
                 OR direct numeric feature vectors (masked retrieval)
    Retrieval  – FAISS vector index (exact or approximate ANN)
                 OR correlation-weighted masked Euclidean distance
    Generation – weighted-mean / local-regression aggregation OR LLM API call

Usage – numeric retrieval + aggregation (recommended, no LLM cost)
-------------------------------------------------------------------
>>> imputer = RAGImputer(n_neighbors=5, mode="aggregation", retrieval="numeric")
>>> imputer.fit(X_train_complete)
>>> X_imputed = imputer.transform(X_missing)

Usage – numeric retrieval + local regression
---------------------------------------------
>>> imputer = RAGImputer(
...     n_neighbors=10,
...     retrieval="numeric",
...     aggregation="local_regression",
... )
>>> imputer.fit(X_train_complete)
>>> X_imputed = imputer.transform(X_missing)

Usage – embedding retrieval + LLM generation
----------------------------------------------
>>> imputer = RAGImputer(
...     n_neighbors=5,
...     mode="llm",
...     retrieval="embedding",
...     llm_model_name="openai/gpt-4.1-nano",
...     llm_api="open_router",
...     dataset_name="Pima Indians Diabetes",
... )
>>> imputer.fit(X_train_complete)
>>> X_imputed = imputer.transform(X_missing)
"""

import re
import warnings
from io import StringIO
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# Optional heavy dependencies – imported lazily so the rest of the module
# remains importable even without sentence-transformers / faiss installed.
# ---------------------------------------------------------------------------

def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding retrieval. "
            "Install it with:  pip install sentence-transformers"
        ) from exc


def _require_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required for embedding retrieval. "
            "Install it with:  pip install faiss-cpu"
        ) from exc


# ---------------------------------------------------------------------------
# Row serialisation helpers
# ---------------------------------------------------------------------------

def _serialize_row(
    values: np.ndarray,
    col_names: list[str],
    mask_nan: np.ndarray | None = None,
) -> str:
    """Convert a numeric row to a natural-language string for embedding.

    Parameters
    ----------
    values :    1-D float array, length n_features
    col_names : feature names, length n_features
    mask_nan :  optional boolean mask (True = missing); those positions are
                omitted from the serialisation (used for query rows)

    Returns
    -------
    str  e.g. "age=0.45, glucose=0.71, bmi=0.38"
    """
    parts = []
    if mask_nan is not None:
        for val, name, is_nan in zip(values, col_names, mask_nan):
            if not is_nan:
                parts.append(f"{name}={val:.4f}")
    else:
        for val, name in zip(values, col_names):
            parts.append(f"{name}={val:.4f}")
    return ", ".join(parts) if parts else "no_observed_features"


def _parse_llm_response(response_text: str, expected_cols: list[str]) -> list[dict]:
    """Extract multiple column→value mappings from a LLM CSV response.

    Returns a list of dicts {col_name: float} for every column found in the response,
    one dict for each row in the CSV.
    """
    match = re.search(r"```(?:csv)?\s*(.*?)\s*```", response_text, re.DOTALL)
    content = match.group(1).strip() if match else response_text.strip()

    for sep in [",", r"\s+"]:
        try:
            df = pd.read_csv(StringIO(content), sep=sep, engine="python")
            if len(df) >= 1:
                results = []
                for idx in range(len(df)):
                    row = df.iloc[idx]
                    result = {}
                    for col in expected_cols:
                        if col in row.index:
                            try:
                                result[col] = float(row[col])
                            except (ValueError, TypeError):
                                pass
                    results.append(result)
                if results:
                    return results
        except Exception:
            continue

    return []


# ---------------------------------------------------------------------------
# LLM prompt builder (RAG-contextualised)
# ---------------------------------------------------------------------------

def _build_rag_prompt(
    dataset_name: str,
    batch_data: list[dict],
    all_cols: list[str],
) -> str:
    """Build a prompt that prepends retrieved context rows before asking the
    LLM to impute the missing values in the query rows (batch processing).
    """
    headers_str = ", ".join(all_cols)
    prompt = f"""
    You are an expert data analyst specializing in the {dataset_name} dataset.
    Your task is to impute missing values in a target record by synthesizing your internal knowledge of the dataset's distributions with the provided reference context.

    SCHEMA:
    - Total Columns: {len(all_cols)}
    - Headers: {headers_str}

    TASK LOGIC:
    1. Analyze the INPUT TO COMPLETE.
    2. Evaluate the REFERENCE DATA provided via retrieval.
    3. Decide: Should the missing value follow the specific pattern of the retrieved neighbors, or the general statistical trend of the {dataset_name} dataset?
    4. Fill all missing values.

    CONSTRAINTS:
    - DO NOT execute Python.
    - DO NOT provide explanations.
    - NO 'NaN', 'None', or '?' values in the output.
    - Maintain the exact column count and order.

    ---
    """
    for i, data in enumerate(batch_data):
        prompt += f"\n[TASK {i+1}]\n"
        prompt += f"CONTEXT (Similar Records): \n{data['context_rows']}\n"
        prompt += f"QUERY (Target Record): {data['missing_row_text']}\n"
        prompt += f"TARGET COLUMNS: {data['missing_cols']}\n"

    prompt += f"""
    OUTPUT FORMAT:
    Return only a CSV code block containing exactly {len(batch_data)} rows.

    ```csv
    {headers_str}
    """
    return prompt


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RAGImputer(BaseEstimator, TransformerMixin):
    """Retrieval-Augmented Generation Imputer with embedding-based retrieval.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbours to retrieve.
    retrieval : {"numeric", "embedding"}, default="numeric"
        Retrieval strategy.
        - ``"numeric"``   – correlation-weighted masked Euclidean distance on
                            observed features only.  No text embedding needed.
                            Recommended for aggregation mode.
        - ``"embedding"`` – sentence-transformer text embeddings + FAISS index.
                            Original approach; useful mainly for LLM mode.
    feature_weighting : {"correlation", "uniform"}, default="correlation"
        How to weight observed features during numeric retrieval.
        - ``"correlation"`` – weight each observed feature by its average
                              absolute correlation with the missing features.
        - ``"uniform"``     – equal weight for all observed features.
    embedding_model : str, default="all-MiniLM-L6-v2"
        Name of the SentenceTransformer model used to embed rows.
        Only loaded when ``retrieval="embedding"`` or ``mode="llm"``.
    faiss_index_type : {"flat", "ivf"}, default="flat"
        FAISS index type (only used when ``retrieval="embedding"``).
        - ``"flat"``  – exact L2 search, best for < 50 k rows.
        - ``"ivf"``   – approximate IVF search, faster for > 50 k rows.
    mode : {"aggregation", "llm"}, default="aggregation"
        Generation strategy after retrieval.
        - ``"aggregation"`` – numeric aggregation of neighbour values.
        - ``"llm"``         – retrieved rows injected as context into an LLM prompt.
    aggregation : {"weighted", "mean", "local_regression"}, default="weighted"
        Aggregation sub-strategy (used only when ``mode="aggregation"``).
        - ``"weighted"``          – inverse-distance weighted mean of neighbours.
        - ``"mean"``              – simple mean of neighbours.
        - ``"local_regression"``  – per-feature Ridge regression fitted on the
                                    retrieved neighbours, using the most
                                    correlated observed features as predictors.
    weights_power : float, default=2.0
        Exponent for inverse-distance weighting.
    llm_model_name : str, default="openai/gpt-4.1-nano"
        LLM model identifier (any model supported by ``algorithms.llm``).
    llm_api : {"open_router","gemini","gpt","claude"}, default="open_router"
        API backend to use when ``mode="llm"``.
    dataset_name : str, default="Unknown Dataset"
        Human-readable dataset name injected into the LLM prompt.
    llm_batch_size : int, default=1
        Number of rows to impute per LLM call when ``mode="llm"``.
    min_complete_rows : int, default=1
        Minimum complete rows required in training data.
    random_state : int or None, default=None
        Reserved for future stochastic variants.

    Attributes
    ----------
    context_store_ : np.ndarray, shape (n_complete, n_features)
        Numeric matrix of complete training rows.
    corr_matrix_ : np.ndarray, shape (n_features, n_features)
        Pearson correlation matrix of the context store features.
    context_texts_ : list[str]
        Serialised text for each context row (used in LLM prompts).
    faiss_index_ : faiss.Index
        Fitted FAISS index (only when retrieval="embedding").
    embedding_model_ : SentenceTransformer
        Loaded embedding model instance (only when needed).
    feature_names_in_ : list[str] or None
    n_features_in_ : int
    """

    _VALID_MODES     = {"aggregation", "llm"}
    _VALID_INDEX     = {"flat", "ivf"}
    _VALID_AGG       = {"weighted", "mean", "local_regression"}
    _VALID_RETRIEVAL = {"numeric", "embedding"}
    _VALID_WEIGHTING = {"correlation", "uniform"}

    def __init__(
        self,
        n_neighbors: int = 5,
        retrieval: Literal["numeric", "embedding"] = "numeric",
        feature_weighting: Literal["correlation", "uniform"] = "correlation",
        embedding_model: str = "all-MiniLM-L6-v2",
        faiss_index_type: Literal["flat", "ivf"] = "flat",
        mode: Literal["aggregation", "llm"] = "aggregation",
        aggregation: Literal["weighted", "mean", "local_regression"] = "weighted",
        weights_power: float = 2.0,
        llm_model_name: str = "openai/gpt-4.1-nano",
        llm_api: Literal["open_router", "gemini", "gpt", "claude"] = "open_router",
        dataset_name: str = "Unknown Dataset",
        llm_batch_size: int = 1,
        min_complete_rows: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.n_neighbors       = n_neighbors
        self.retrieval         = retrieval
        self.feature_weighting = feature_weighting
        self.embedding_model   = embedding_model
        self.faiss_index_type  = faiss_index_type
        self.mode              = mode
        self.aggregation       = aggregation
        self.weights_power     = weights_power
        self.llm_model_name    = llm_model_name
        self.llm_api           = llm_api
        self.dataset_name      = dataset_name
        self.llm_batch_size    = llm_batch_size
        self.min_complete_rows = min_complete_rows
        self.random_state      = random_state

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_params(self) -> None:
        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 1:
            raise ValueError("`n_neighbors` must be a positive integer.")
        if self.mode not in self._VALID_MODES:
            raise ValueError(f"`mode` must be one of {self._VALID_MODES}.")
        if self.faiss_index_type not in self._VALID_INDEX:
            raise ValueError(f"`faiss_index_type` must be one of {self._VALID_INDEX}.")
        if self.aggregation not in self._VALID_AGG:
            raise ValueError(f"`aggregation` must be one of {self._VALID_AGG}.")
        if self.retrieval not in self._VALID_RETRIEVAL:
            raise ValueError(f"`retrieval` must be one of {self._VALID_RETRIEVAL}.")
        if self.feature_weighting not in self._VALID_WEIGHTING:
            raise ValueError(f"`feature_weighting` must be one of {self._VALID_WEIGHTING}.")
        if self.weights_power <= 0:
            raise ValueError("`weights_power` must be positive.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float, na_value=np.nan)
        return np.array(X, dtype=float)

    def _get_col_names(self, n_features: int) -> list[str]:
        if self.feature_names_in_ is not None:
            return list(self.feature_names_in_)
        return [f"x{i}" for i in range(n_features)]

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings → float32 matrix (n, d)."""
        vecs = self.embedding_model_.encode(
            texts,
            batch_size=128,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,   # cosine-equivalent L2 on unit sphere
        )
        return vecs.astype(np.float32)

    def _build_index(self, vectors: np.ndarray):
        """Create and populate a FAISS index."""
        faiss = _require_faiss()
        d = vectors.shape[1]

        if self.faiss_index_type == "flat" or len(vectors) < 1000:
            index = faiss.IndexFlatIP(d)
        else:  # ivf – approximate, better for large datasets
            nlist = min(int(np.sqrt(len(vectors))), 256)
            quantiser = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantiser, d, nlist)
            index.train(vectors)

        index.add(vectors)
        return index

    # ------------------------------------------------------------------
    # Numeric retrieval (correlation-weighted masked distance)
    # ------------------------------------------------------------------

    def _retrieve_numeric(
        self,
        query_row: np.ndarray,
        missing_mask: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve k nearest neighbours using masked, correlation-weighted
        Euclidean distance on observed features only.

        For each query row, only the observed (non-missing) features are used
        to compute distances.  When ``feature_weighting="correlation"``, each
        observed feature is weighted by its average absolute Pearson correlation
        with the missing features — so features that are predictive of the
        target receive higher influence during retrieval.

        Returns
        -------
        indices : np.ndarray, shape (actual_k,)
            Indices into ``context_store_``.
        distances : np.ndarray, shape (actual_k,)
            Weighted Euclidean distances (lower = closer).
        """
        observed_mask = ~missing_mask
        observed_idx = np.where(observed_mask)[0]
        missing_idx = np.where(missing_mask)[0]

        # --- Feature weights for observed dimensions ---
        if (
            self.feature_weighting == "correlation"
            and hasattr(self, "corr_matrix_")
            and len(missing_idx) > 0
        ):
            # Average |correlation| of each observed feature with all missing features
            w = np.abs(
                self.corr_matrix_[np.ix_(observed_idx, missing_idx)]
            ).mean(axis=1)
            # Small baseline prevents zeroing out uncorrelated features
            w = w + 0.05
            w = w / w.sum()
        else:
            n_obs = max(len(observed_idx), 1)
            w = np.ones(n_obs) / n_obs

        # --- Weighted Euclidean distance ---
        context_obs = self.context_store_[:, observed_idx]
        query_obs = query_row[observed_idx]
        diffs = context_obs - query_obs
        sq_dists = (diffs ** 2) @ w   # (n_context,)

        # --- Top-k selection ---
        n = len(sq_dists)
        actual_k = min(k, n)
        if actual_k >= n:
            order = np.argsort(sq_dists)
        else:
            part = np.argpartition(sq_dists, actual_k)[:actual_k]
            order = part[np.argsort(sq_dists[part])]

        return order, np.sqrt(sq_dists[order])

    # ------------------------------------------------------------------
    # Embedding retrieval (FAISS)
    # ------------------------------------------------------------------

    def _retrieve_embedding(self, query_vec: np.ndarray, k: int):
        """FAISS search. Returns (indices, distances)."""
        q = query_vec.reshape(1, -1).astype(np.float32)
        distances, indices = self.faiss_index_.search(q, k)
        return indices[0], distances[0]

    # ------------------------------------------------------------------
    # Aggregation strategies
    # ------------------------------------------------------------------

    def _aggregate_neighbours(
        self,
        neighbours: np.ndarray,
        l2_distances: np.ndarray,
        missing_mask: np.ndarray,
        original_row: np.ndarray,
    ) -> np.ndarray:
        """Weighted-mean, mean, or local-regression aggregation."""
        result = original_row.copy()
        if self.aggregation == "mean":
            result[missing_mask] = neighbours[:, missing_mask].mean(axis=0)
        elif self.aggregation == "local_regression":
            result = self._aggregate_local_regression(
                neighbours, missing_mask, original_row,
            )
        else:  # weighted
            eps = 1e-10
            weights = 1.0 / (l2_distances ** self.weights_power + eps)
            weights /= weights.sum()
            result[missing_mask] = (
                neighbours[:, missing_mask] * weights[:, np.newaxis]
            ).sum(axis=0)
        return result

    def _aggregate_local_regression(
        self,
        neighbours: np.ndarray,
        missing_mask: np.ndarray,
        query_row: np.ndarray,
    ) -> np.ndarray:
        """Fit a per-feature Ridge regression on retrieved neighbours.

        For each missing feature j, fits a small Ridge model predicting j from
        the most correlated observed features.  The number of predictors is
        capped at ``len(neighbours) - 1`` to avoid overfitting.
        """
        from sklearn.linear_model import Ridge

        result = query_row.copy()
        observed_idx = np.where(~missing_mask)[0]
        missing_idx = np.where(missing_mask)[0]

        if len(observed_idx) == 0 or len(neighbours) < 2:
            result[missing_mask] = neighbours[:, missing_mask].mean(axis=0)
            return result

        X_local = neighbours[:, observed_idx]
        query_obs = query_row[observed_idx].reshape(1, -1)

        # Cap predictors to avoid overfitting with few neighbours
        max_predictors = max(1, len(neighbours) - 1)

        for j in missing_idx:
            y_local = neighbours[:, j]

            # Constant target → just use the mean
            if np.std(y_local) < 1e-10:
                result[j] = y_local.mean()
                continue

            # Select most correlated observed features if too many
            if len(observed_idx) > max_predictors and hasattr(self, "corr_matrix_"):
                corrs = np.abs(self.corr_matrix_[observed_idx, j])
                top_feat = np.argsort(corrs)[-max_predictors:]
                X_sub = X_local[:, top_feat]
                q_sub = query_obs[:, top_feat]
            else:
                X_sub = X_local
                q_sub = query_obs

            reg = Ridge(alpha=1.0)
            reg.fit(X_sub, y_local)
            result[j] = reg.predict(q_sub)[0]

        return result

    # ------------------------------------------------------------------
    # LLM helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _llm_impute_batch(
        self,
        batch_rows: list[np.ndarray],
        batch_missing_masks: list[np.ndarray],
        batch_neighbour_indices: list[np.ndarray],
        col_names: list[str],
    ) -> list[np.ndarray]:
        """Call LLM with retrieved context to impute a batch of rows."""
        batch_data = []
        for row, missing_mask, neighbour_indices in zip(batch_rows, batch_missing_masks, batch_neighbour_indices):
            context_texts = [self.context_texts_[i] for i in neighbour_indices]
            query_text = _serialize_row(row, col_names, mask_nan=missing_mask)
            missing_cols = [col_names[i] for i in np.where(missing_mask)[0]]
            batch_data.append({
                "missing_row_text": query_text,
                "context_rows": context_texts,
                "missing_cols": missing_cols,
            })

        prompt = _build_rag_prompt(
            dataset_name=self.dataset_name,
            batch_data=batch_data,
            all_cols=col_names,
        )

        # Call the LLM via the existing llm module
        response_text = self._call_llm(prompt)

        # Parse the response
        imputed_vals_list = _parse_llm_response(response_text, col_names)

        results = []
        for i, (row, missing_mask, neighbour_indices) in enumerate(zip(batch_rows, batch_missing_masks, batch_neighbour_indices)):
            result = row.copy()
            imputed_vals = imputed_vals_list[i] if i < len(imputed_vals_list) else {}

            for j, col in enumerate(col_names):
                if missing_mask[j]:
                    if col in imputed_vals:
                        result[j] = imputed_vals[col]
                    else:
                        # Fallback: use mean of context neighbours
                        context_arr = self.context_store_[neighbour_indices]
                        result[j] = context_arr[:, j].mean()
                        warnings.warn(
                            f"LLM did not return value for '{col}' in row {i} of batch. "
                            "Using neighbour mean as fallback.",
                            UserWarning,
                            stacklevel=4,
                        )
            results.append(result)

        return results

    def _call_llm(self, prompt: str) -> str:
        """Dispatch a single prompt to the configured LLM backend."""
        import os
        from dotenv import load_dotenv

        load_dotenv()

        match self.llm_api:
            case "open_router":
                from openai import OpenAI
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("API_KEY_OPEN_ROUTER"),
                )
                response = client.responses.create(
                    model=self.llm_model_name,
                    temperature=0.05,
                    input=prompt,
                )
                return response.output[0].content[0].text

            case "gpt":
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("API_KEY_GPT"))
                response = client.responses.create(
                    model=self.llm_model_name,
                    input=prompt,
                )
                return response.output_text

            case "gemini":
                from google import genai
                from google.genai import types
                client = genai.Client(
                    api_key=os.getenv("API_KEY_GEMINI"),
                    http_options={"timeout": 10 * 60 * 1000},
                )
                response = client.models.generate_content(
                    model=self.llm_model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.05,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                return response.text.strip()

            case "claude":
                import anthropic
                client = anthropic.Anthropic(api_key=os.getenv("API_KEY_CLAUDE"))
                response = client.messages.create(
                    model=self.llm_model_name,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.05,
                )
                return response.content[0].text

            case _:
                raise ValueError(f"Unknown LLM api: '{self.llm_api}'")

    # ------------------------------------------------------------------
    # Public API – sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X, y=None) -> "RAGImputer":
        """Build the context store from complete rows.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Training data. Only fully complete rows are retained.
        y : ignored

        Returns
        -------
        self
        """
        self._validate_params()

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_: list[str] | None = list(X.columns)
        else:
            self.feature_names_in_ = None

        X_arr = self._to_numpy(X)
        self.n_features_in_: int = X_arr.shape[1]
        col_names = self._get_col_names(self.n_features_in_)

        # Filter to complete rows only
        complete_mask = ~np.isnan(X_arr).any(axis=1)
        context = X_arr[complete_mask]

        if len(context) < self.min_complete_rows:
            raise ValueError(
                f"Only {len(context)} complete row(s) found, but "
                f"`min_complete_rows={self.min_complete_rows}` required."
            )
        if len(context) < self.n_neighbors:
            warnings.warn(
                f"Context store has {len(context)} complete rows, fewer than "
                f"n_neighbors={self.n_neighbors}. All rows will be used.",
                UserWarning,
                stacklevel=2,
            )

        # Store raw numeric context
        self.context_store_: np.ndarray = context

        # Compute feature correlation matrix (used by numeric retrieval
        # and local_regression aggregation)
        if (
            self.feature_weighting == "correlation"
            or self.aggregation == "local_regression"
        ):
            self.corr_matrix_: np.ndarray = np.corrcoef(context.T)
            if self.corr_matrix_.ndim == 0:
                self.corr_matrix_ = np.array([[1.0]])
            np.nan_to_num(self.corr_matrix_, copy=False, nan=0.0)

        # Serialise rows to text (always needed for LLM prompts; kept for
        # embedding retrieval path as well)
        context_texts = [
            _serialize_row(row, col_names) for row in context
        ]
        self.context_texts_: list[str] = context_texts

        # Build embedding index only when needed
        if self.retrieval == "embedding" or self.mode == "llm":
            SentenceTransformer = _require_sentence_transformers()
            self.embedding_model_ = SentenceTransformer(
                self.embedding_model, cache_folder="./embedding_cache",
            )
            context_vecs = self._embed(context_texts)
            self.faiss_index_ = self._build_index(context_vecs)

        return self

    def transform(self, X, y=None) -> np.ndarray:
        """Impute missing values using retrieval-augmented generation.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Data with missing values.

        Returns
        -------
        X_imputed : np.ndarray, shape (n_samples, n_features)
        """
        check_is_fitted(self, ["context_store_"])

        X_arr = self._to_numpy(X).copy()

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features but the imputer was fitted "
                f"with {self.n_features_in_} features."
            )

        col_names = self._get_col_names(self.n_features_in_)
        k = min(self.n_neighbors, len(self.context_store_))

        use_numeric = (self.retrieval == "numeric")

        # ── Identify rows that need imputation ─────────────────────────
        missing_row_indices = []
        for i, row in enumerate(X_arr):
            missing_mask = np.isnan(row)
            if not missing_mask.any():
                continue

            # Fully-missing row fallback
            if (~missing_mask).sum() == 0:
                X_arr[i] = np.nanmean(self.context_store_, axis=0)
                warnings.warn(
                    f"Row {i} is entirely missing. "
                    "Filled with context column means.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            missing_row_indices.append(i)

        if not missing_row_indices:
            return X_arr

        # ==============================================================
        # NUMERIC RETRIEVAL PATH
        # ==============================================================
        if use_numeric:
            if self.mode == "aggregation":
                for idx in missing_row_indices:
                    row = X_arr[idx]
                    missing_mask = np.isnan(row)
                    nn_idx, nn_dists = self._retrieve_numeric(
                        row, missing_mask, k,
                    )
                    neighbours = self.context_store_[nn_idx]
                    X_arr[idx] = self._aggregate_neighbours(
                        neighbours, nn_dists, missing_mask, row,
                    )
            else:  # "llm"
                batch_indices: list[int] = []
                batch_rows: list[np.ndarray] = []
                batch_missing_masks: list[np.ndarray] = []
                batch_nn_indices: list[np.ndarray] = []

                for idx in missing_row_indices:
                    row = X_arr[idx]
                    missing_mask = np.isnan(row)
                    nn_idx, _ = self._retrieve_numeric(
                        row, missing_mask, k,
                    )

                    batch_indices.append(idx)
                    batch_rows.append(row)
                    batch_missing_masks.append(missing_mask)
                    batch_nn_indices.append(nn_idx)

                    if len(batch_indices) == self.llm_batch_size:
                        imputed = self._llm_impute_batch(
                            batch_rows, batch_missing_masks,
                            batch_nn_indices, col_names,
                        )
                        for bi, ir in zip(batch_indices, imputed):
                            X_arr[bi] = ir
                        batch_indices.clear()
                        batch_rows.clear()
                        batch_missing_masks.clear()
                        batch_nn_indices.clear()

                if batch_indices:
                    imputed = self._llm_impute_batch(
                        batch_rows, batch_missing_masks,
                        batch_nn_indices, col_names,
                    )
                    for bi, ir in zip(batch_indices, imputed):
                        X_arr[bi] = ir

        # ==============================================================
        # EMBEDDING RETRIEVAL PATH (original approach)
        # ==============================================================
        else:
            check_is_fitted(self, ["faiss_index_", "embedding_model_"])

            # Batch-embed all query texts
            query_texts = []
            for idx in missing_row_indices:
                row = X_arr[idx]
                missing_mask = np.isnan(row)
                query_texts.append(
                    _serialize_row(row, col_names, mask_nan=missing_mask)
                )

            query_vecs = self._embed(query_texts)
            distances, indices = self.faiss_index_.search(query_vecs, k)

            if self.mode == "aggregation":
                for idx, nn_idx, nn_dists in zip(
                    missing_row_indices, indices, distances,
                ):
                    row = X_arr[idx]
                    missing_mask = np.isnan(row)
                    neighbours = self.context_store_[nn_idx]
                    X_arr[idx] = self._aggregate_neighbours(
                        neighbours, nn_dists, missing_mask, row,
                    )
            else:  # "llm"
                batch_indices = []
                batch_rows = []
                batch_missing_masks = []
                batch_nn_indices = []

                for idx, nn_idx in zip(missing_row_indices, indices):
                    row = X_arr[idx]
                    missing_mask = np.isnan(row)

                    batch_indices.append(idx)
                    batch_rows.append(row)
                    batch_missing_masks.append(missing_mask)
                    batch_nn_indices.append(nn_idx)

                    if len(batch_indices) == self.llm_batch_size:
                        imputed = self._llm_impute_batch(
                            batch_rows, batch_missing_masks,
                            batch_nn_indices, col_names,
                        )
                        for bi, ir in zip(batch_indices, imputed):
                            X_arr[bi] = ir
                        batch_indices.clear()
                        batch_rows.clear()
                        batch_missing_masks.clear()
                        batch_nn_indices.clear()

                if batch_indices:
                    imputed = self._llm_impute_batch(
                        batch_rows, batch_missing_masks,
                        batch_nn_indices, col_names,
                    )
                    for bi, ir in zip(batch_indices, imputed):
                        X_arr[bi] = ir

        return X_arr

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step (uses complete rows of X as context)."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        check_is_fitted(self, "context_store_")
        return self._get_col_names(self.n_features_in_)
