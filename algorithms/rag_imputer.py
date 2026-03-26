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
    Retrieval  – FAISS vector index (exact or approximate ANN)
    Generation – weighted-mean aggregation  OR  LLM API call with context

Usage – aggregation mode (no LLM cost)
---------------------------------------
>>> imputer = RAGImputer(n_neighbors=5, mode="aggregation")
>>> imputer.fit(X_train_complete)
>>> X_imputed = imputer.transform(X_missing)

Usage – LLM generation mode
----------------------------
>>> imputer = RAGImputer(
...     n_neighbors=5,
...     mode="llm",
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
            "sentence-transformers is required for RAGImputer. "
            "Install it with:  pip install sentence-transformers"
        ) from exc


def _require_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required for RAGImputer. "
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
    # Refined logic to prevent "breaking"
    prompt = f"ACT AS A DATA CONVERSION ENGINE. Task: Impute missing values for {dataset_name}.\n"
    prompt += f"SCHEMA: Total of {len(all_cols)} columns. Headers: {headers_str}\n"

    for i, data in enumerate(batch_data):
        prompt += f"\n--- TASK {i+1} ---\n"
        prompt += f"REFERENCE DATA (Context):\n{data['context_rows']}\n"
        prompt += f"INPUT TO COMPLETE (Query): {data['missing_row_text']}\n"
        prompt += f"FILL THESE SPECIFIC COLUMNS: {data['missing_cols']}\n"

    prompt += f"""
    ---
    OUTPUT INSTRUCTIONS:
    1. Provide exactly {len(batch_data)} rows of data.
    2. DO NOT change existing values in the Query.
    3. Every row must have {len(all_cols)} comma-separated values.
    4. Output MUST be a single code block.

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
        Number of nearest neighbours to retrieve from the FAISS index.
    embedding_model : str, default="all-MiniLM-L6-v2"
        Name of the SentenceTransformer model used to embed rows.
        Any model from https://huggingface.co/sentence-transformers is accepted.
    faiss_index_type : {"flat", "ivf"}, default="flat"
        FAISS index type.
        - ``"flat"``  – exact L2 search, best for < 50 k rows.
        - ``"ivf"``   – approximate IVF search, faster for > 50 k rows.
    mode : {"aggregation", "llm"}, default="aggregation"
        Generation strategy after retrieval.
        - ``"aggregation"`` – inverse-distance weighted mean of neighbour values.
        - ``"llm"``         – retrieved rows injected as context into an LLM prompt.
    aggregation : {"weighted", "mean"}, default="weighted"
        Aggregation sub-strategy (used only when ``mode="aggregation"``).
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
        Setting > 1 reduces API round-trips but increases prompt size.
    min_complete_rows : int, default=1
        Minimum complete rows required in training data.
    random_state : int or None, default=None
        Reserved for future stochastic variants.

    Attributes
    ----------
    context_store_ : np.ndarray, shape (n_complete, n_features)
        Numeric matrix of complete training rows (retrieved for aggregation).
    context_texts_ : list[str]
        Serialised text for each context row (used in LLM prompts).
    faiss_index_ : faiss.Index
        Fitted FAISS index.
    embedding_model_ : SentenceTransformer
        Loaded embedding model instance.
    feature_names_in_ : list[str] or None
    n_features_in_ : int
    """

    _VALID_MODES = {"aggregation", "llm"}
    _VALID_INDEX = {"flat", "ivf"}
    _VALID_AGG   = {"weighted", "mean"}

    def __init__(
        self,
        n_neighbors: int = 5,
        embedding_model: str = "all-MiniLM-L6-v2",
        faiss_index_type: Literal["flat", "ivf"] = "flat",
        mode: Literal["aggregation", "llm"] = "aggregation",
        aggregation: Literal["weighted", "mean"] = "weighted",
        weights_power: float = 2.0,
        llm_model_name: str = "openai/gpt-4.1-nano",
        llm_api: Literal["open_router", "gemini", "gpt", "claude"] = "open_router",
        dataset_name: str = "Unknown Dataset",
        llm_batch_size: int = 1,
        min_complete_rows: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.n_neighbors       = n_neighbors
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
            index = faiss.IndexFlatL2(d)
        else:  # ivf – approximate, better for large datasets
            nlist = min(int(np.sqrt(len(vectors))), 256)
            quantiser = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantiser, d, nlist)
            index.train(vectors)

        index.add(vectors)
        return index

    def _retrieve(self, query_vec: np.ndarray, k: int):
        """FAISS search. Returns (indices, l2_distances)."""
        q = query_vec.reshape(1, -1).astype(np.float32)
        distances, indices = self.faiss_index_.search(q, k)
        return indices[0], distances[0]

    def _aggregate_neighbours(
        self,
        neighbours: np.ndarray,
        l2_distances: np.ndarray,
        missing_mask: np.ndarray,
        original_row: np.ndarray,
    ) -> np.ndarray:
        """Weighted-mean or mean aggregation over retrieved rows."""
        result = original_row.copy()
        if self.aggregation == "mean":
            result[missing_mask] = neighbours[:, missing_mask].mean(axis=0)
        else:
            eps = 1e-10
            weights = 1.0 / (l2_distances ** self.weights_power + eps)
            weights /= weights.sum()
            result[missing_mask] = (
                neighbours[:, missing_mask] * weights[:, np.newaxis]
            ).sum(axis=0)
        return result

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
        """Build the embedding-based FAISS context store from complete rows.

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

        # Serialise rows to text
        context_texts = [
            _serialize_row(row, col_names) for row in context
        ]

        # Load embedding model
        SentenceTransformer = _require_sentence_transformers()
        self.embedding_model_ = SentenceTransformer(self.embedding_model, cache_folder="./embedding_cache")

        # Embed and build FAISS index
        context_vecs = self._embed(context_texts)   # (n_complete, d_embed)
        self.faiss_index_ = self._build_index(context_vecs)

        # Store raw numeric context and text for generation
        self.context_store_: np.ndarray = context
        self.context_texts_: list[str] = context_texts

        return self

    def transform(self, X, y=None) -> np.ndarray:
        """Impute missing values using embedding-based retrieval.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Data with missing values.

        Returns
        -------
        X_imputed : np.ndarray, shape (n_samples, n_features)
        """
        check_is_fitted(self, ["context_store_", "faiss_index_", "embedding_model_"])

        X_arr = self._to_numpy(X).copy()

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features but the imputer was fitted "
                f"with {self.n_features_in_} features."
            )

        col_names = self._get_col_names(self.n_features_in_)
        k = min(self.n_neighbors, len(self.context_store_))

        batch_indices = []
        batch_rows = []
        batch_missing_masks = []
        batch_nn_indices = []
        
        valid_missing_indices = []
        query_texts = []

        # ── Pass 1: Handle complete/fully-missing rows and gather query texts ─
        for i, row in enumerate(X_arr):
            missing_mask = np.isnan(row)
            if not missing_mask.any():
                continue  # row is complete

            observed_mask = ~missing_mask

            # ── Fully-missing row fallback ────────────────────────────
            if not observed_mask.any():
                X_arr[i] = np.nanmean(self.context_store_, axis=0)
                warnings.warn(
                    f"Row {i} is entirely missing. "
                    "Filled with context column means.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            query_text = _serialize_row(row, col_names, mask_nan=missing_mask)
            query_texts.append(query_text)
            valid_missing_indices.append(i)

        if not valid_missing_indices:
            return X_arr

        # ── Pass 2: Batched embedding and FAISS retrieval ─────────────────────
        query_vecs = self._embed(query_texts)
        distances, indices = self.faiss_index_.search(query_vecs, k)

        # ── Pass 3: Generation (Aggregation or LLM) ───────────────────────────
        if self.mode == "aggregation":
            for idx, nn_indices, nn_dists in zip(valid_missing_indices, indices, distances):
                row = X_arr[idx]
                missing_mask = np.isnan(row)
                neighbours = self.context_store_[nn_indices]
                X_arr[idx] = self._aggregate_neighbours(
                    neighbours, nn_dists, missing_mask, row
                )
        else:  # "llm"
            for idx, nn_indices in zip(valid_missing_indices, indices):
                row = X_arr[idx]
                missing_mask = np.isnan(row)

                batch_indices.append(idx)
                batch_rows.append(row)
                batch_missing_masks.append(missing_mask)
                batch_nn_indices.append(nn_indices)

                if len(batch_indices) == self.llm_batch_size:
                    imputed_rows = self._llm_impute_batch(
                        batch_rows, batch_missing_masks, batch_nn_indices, col_names
                    )
                    for i_batch, imp_row in zip(batch_indices, imputed_rows):
                        X_arr[i_batch] = imp_row
                    
                    batch_indices.clear()
                    batch_rows.clear()
                    batch_missing_masks.clear()
                    batch_nn_indices.clear()

            if batch_indices:  # process remaining
                imputed_rows = self._llm_impute_batch(
                    batch_rows, batch_missing_masks, batch_nn_indices, col_names
                )
                for i_batch, imp_row in zip(batch_indices, imputed_rows):
                    X_arr[i_batch] = imp_row

        return X_arr

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step (uses complete rows of X as context)."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        check_is_fitted(self, "context_store_")
        return self._get_col_names(self.n_features_in_)
