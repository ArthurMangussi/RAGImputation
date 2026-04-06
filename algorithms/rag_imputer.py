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

Retrieval uses correlation-weighted masked Euclidean distance on the
observed features of each query row.  Retrieved neighbours are serialised
as context in an LLM prompt that produces the imputed values.

Usage
-----
>>> imputer = RAGImputer(
...     n_neighbors=5,
...     llm_model_name="gemini-3-flash-preview",
...     llm_api="gemini",
...     dataset_name="Pima Indians Diabetes",
... )
>>> imputer.fit(X_train)
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
# Row serialisation
# ---------------------------------------------------------------------------

def _serialize_row(
    values: np.ndarray,
    col_names: list[str],
    mask_nan: np.ndarray | None = None,
) -> str:
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


def _build_rag_prompt(
    dataset_name: str,
    batch_data: list[dict],
    all_cols: list[str],
) -> str:
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
    """Retrieval-Augmented Generation Imputer.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbours to retrieve.
    feature_weighting : {"correlation", "uniform"}, default="correlation"
        How to weight observed features during retrieval.
        - ``"correlation"`` – weight by |correlation| with missing features.
        - ``"uniform"``     – equal weight for all observed features.
    llm_model_name : str, default="openai/gpt-4.1-nano"
        LLM model identifier.
    llm_api : {"open_router","gemini","gpt","claude"}, default="open_router"
        API backend for LLM calls.
    dataset_name : str, default="Unknown Dataset"
        Injected into the LLM prompt.
    llm_batch_size : int, default=1
        Rows per LLM call.
    min_complete_rows : int, default=1
        Minimum complete rows required in training data.
    """

    _VALID_WEIGHTING = {"correlation", "uniform"}

    def __init__(
        self,
        n_neighbors: int = 5,
        feature_weighting: Literal["correlation", "uniform"] = "correlation",
        llm_model_name: str = "openai/gpt-4.1-nano",
        llm_api: Literal["open_router", "gemini", "gpt", "claude"] = "open_router",
        dataset_name: str = "Unknown Dataset",
        llm_batch_size: int = 1,
        min_complete_rows: int = 1,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.feature_weighting = feature_weighting
        self.llm_model_name = llm_model_name
        self.llm_api = llm_api
        self.dataset_name = dataset_name
        self.llm_batch_size = llm_batch_size
        self.min_complete_rows = min_complete_rows

    def _validate_params(self) -> None:
        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 1:
            raise ValueError("`n_neighbors` must be a positive integer.")
        if self.feature_weighting not in self._VALID_WEIGHTING:
            raise ValueError(
                f"`feature_weighting` must be one of {self._VALID_WEIGHTING}."
            )

    # ------------------------------------------------------------------
    # Helpers
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

    # ------------------------------------------------------------------
    # Retrieval: correlation-weighted masked Euclidean distance
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        query_row: np.ndarray,
        missing_mask: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve k nearest neighbours using only the observed features,
        weighted by their correlation with the missing features."""
        observed_idx = np.where(~missing_mask)[0]
        missing_idx = np.where(missing_mask)[0]

        if self.feature_weighting == "correlation" and len(missing_idx) > 0:
            w = np.abs(self.corr_matrix_[np.ix_(observed_idx, missing_idx)]).mean(
                axis=1
            )
            w = w + 0.05
            w = w / w.sum()
        else:
            n_obs = max(len(observed_idx), 1)
            w = np.ones(n_obs) / n_obs

        diffs = self.context_store_[:, observed_idx] - query_row[observed_idx]
        sq_dists = (diffs ** 2) @ w

        n = len(sq_dists)
        actual_k = min(k, n)
        if actual_k >= n:
            order = np.argsort(sq_dists)
        else:
            part = np.argpartition(sq_dists, actual_k)[:actual_k]
            order = part[np.argsort(sq_dists[part])]

        return order, np.sqrt(sq_dists[order])

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------

    def _llm_impute_batch(
        self,
        batch_rows: list[np.ndarray],
        batch_missing_masks: list[np.ndarray],
        batch_neighbour_indices: list[np.ndarray],
        col_names: list[str],
    ) -> list[np.ndarray]:
        batch_data = []
        for row, mask, nn_idx in zip(
            batch_rows, batch_missing_masks, batch_neighbour_indices
        ):
            context_texts = [self.context_texts_[i] for i in nn_idx]
            batch_data.append(
                {
                    "missing_row_text": _serialize_row(row, col_names, mask_nan=mask),
                    "context_rows": context_texts,
                    "missing_cols": [col_names[i] for i in np.where(mask)[0]],
                }
            )

        response_text = self._call_llm(
            _build_rag_prompt(self.dataset_name, batch_data, col_names)
        )
        imputed_vals_list = _parse_llm_response(response_text, col_names)

        results = []
        for i, (row, mask, nn_idx) in enumerate(
            zip(batch_rows, batch_missing_masks, batch_neighbour_indices)
        ):
            result = row.copy()
            vals = imputed_vals_list[i] if i < len(imputed_vals_list) else {}

            for j, col in enumerate(col_names):
                if mask[j]:
                    if col in vals:
                        result[j] = vals[col]
                    else:
                        result[j] = self.context_store_[nn_idx][:, j].mean()
                        warnings.warn(
                            f"LLM did not return '{col}' in row {i}. "
                            "Using neighbour mean.",
                            UserWarning,
                            stacklevel=4,
                        )
            results.append(result)

        return results

    def _call_llm(self, prompt: str) -> str:
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
                resp = client.responses.create(
                    model=self.llm_model_name,
                    temperature=0.05,
                    input=prompt,
                )
                return resp.output[0].content[0].text

            case "gpt":
                from openai import OpenAI

                client = OpenAI(api_key=os.getenv("API_KEY_GPT"))
                resp = client.responses.create(
                    model=self.llm_model_name,
                    input=prompt,
                )
                return resp.output_text

            case "gemini":
                from google import genai
                from google.genai import types

                client = genai.Client(
                    api_key=os.getenv("API_KEY_GEMINI"),
                    http_options={"timeout": 10 * 60 * 1000},
                )
                resp = client.models.generate_content(
                    model=self.llm_model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.05,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                return resp.text.strip()

            case "claude":
                import anthropic

                client = anthropic.Anthropic(api_key=os.getenv("API_KEY_CLAUDE"))
                resp = client.messages.create(
                    model=self.llm_model_name,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.05,
                )
                return resp.content[0].text

            case _:
                raise ValueError(f"Unknown LLM api: '{self.llm_api}'")

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X, y=None) -> "RAGImputer":
        self._validate_params()

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_: list[str] | None = list(X.columns)
        else:
            self.feature_names_in_ = None

        X_arr = self._to_numpy(X)
        self.n_features_in_: int = X_arr.shape[1]
        col_names = self._get_col_names(self.n_features_in_)

        # Keep only complete rows
        complete_mask = ~np.isnan(X_arr).any(axis=1)
        context = X_arr[complete_mask]

        if len(context) < self.min_complete_rows:
            raise ValueError(
                f"Only {len(context)} complete row(s) found, need "
                f"{self.min_complete_rows}."
            )
        if len(context) < self.n_neighbors:
            warnings.warn(
                f"Context has {len(context)} rows < n_neighbors={self.n_neighbors}.",
                UserWarning,
                stacklevel=2,
            )

        self.context_store_: np.ndarray = context

        # Correlation matrix for retrieval weights
        self.corr_matrix_: np.ndarray = np.corrcoef(context.T)
        if self.corr_matrix_.ndim == 0:
            self.corr_matrix_ = np.array([[1.0]])
        np.nan_to_num(self.corr_matrix_, copy=False, nan=0.0)

        # Text representations for LLM prompts
        self.context_texts_: list[str] = [
            _serialize_row(row, col_names) for row in context
        ]

        return self

    def transform(self, X, y=None) -> np.ndarray:
        check_is_fitted(self, ["context_store_"])

        X_arr = self._to_numpy(X).copy()
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}."
            )

        col_names = self._get_col_names(self.n_features_in_)
        k = min(self.n_neighbors, len(self.context_store_))

        # Identify rows needing imputation
        missing_row_indices = []
        for i, row in enumerate(X_arr):
            mask = np.isnan(row)
            if not mask.any():
                continue
            if (~mask).sum() == 0:
                X_arr[i] = np.nanmean(self.context_store_, axis=0)
                warnings.warn(
                    f"Row {i} entirely missing. Filled with column means.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            missing_row_indices.append(i)

        if not missing_row_indices:
            return X_arr

        batch_idx: list[int] = []
        batch_rows: list[np.ndarray] = []
        batch_masks: list[np.ndarray] = []
        batch_nn: list[np.ndarray] = []

        for idx in missing_row_indices:
            row = X_arr[idx]
            mask = np.isnan(row)
            nn_idx, _ = self._retrieve(row, mask, k)

            batch_idx.append(idx)
            batch_rows.append(row)
            batch_masks.append(mask)
            batch_nn.append(nn_idx)

            if len(batch_idx) == self.llm_batch_size:
                for bi, ir in zip(
                    batch_idx,
                    self._llm_impute_batch(
                        batch_rows, batch_masks, batch_nn, col_names
                    ),
                ):
                    X_arr[bi] = ir
                batch_idx.clear()
                batch_rows.clear()
                batch_masks.clear()
                batch_nn.clear()

        if batch_idx:
            for bi, ir in zip(
                batch_idx,
                self._llm_impute_batch(batch_rows, batch_masks, batch_nn, col_names),
            ):
                X_arr[bi] = ir

        return X_arr

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        check_is_fitted(self, "context_store_")
        return self._get_col_names(self.n_features_in_)
