# TabRAG-Imputer

> **Retrieval-Augmented Generation for Missing Data Imputation in Tabular Data**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn compatible](https://img.shields.io/badge/sklearn-compatible-orange.svg)](https://scikit-learn.org/)

**TabRAG-Imputer** is a novel imputation framework that combines the structural awareness of correlation-weighted retrieval with the generative capabilities of Large Language Models (LLMs). Rather than relying solely on statistical distance (like KNN) or parametric knowledge (like zero-shot LLMs), TabRAG-Imputer retrieves the most relevant complete records from your dataset and feeds them as grounded context to an LLM, enabling accurate, dataset-specific imputation.

Evaluated across 20 datasets under MAR and MNAR mechanisms, TabRAG-Imputer achieves the **lowest overall MAE under MNAR** and **ranks second under MAR**, trailing only MICE.

---

## How it works

TabRAG-Imputer operates in three stages for each incomplete row:

```
Incomplete row
      │
      ▼
┌─────────────────────────────┐
│  1. Correlation-weighted    │  Ranks observed features by their
│     context retrieval       │  correlation with missing features,
│                             │  then retrieves the k most similar
│                             │  complete rows via weighted distance
└─────────────┬───────────────┘
              │  k complete rows
              ▼
┌─────────────────────────────┐
│  2. Row serialisation       │  Converts retrieved rows and the
│                             │  incomplete query into structured
│                             │  key=value text representations
└─────────────┬───────────────┘
              │  Structured prompt
              ▼
┌─────────────────────────────┐
│  3. LLM-based generation    │  LLM predicts missing values using
│                             │  retrieved context as grounding;
│                             │  output enforced as CSV for parsing
└─────────────────────────────┘
```

### Retrieval mechanism

For a query row **x*** with observed features *O* and missing features *M*, each observed feature *j* is weighted by its average absolute Pearson correlation with the missing features:

$$\bar{r}_j = \frac{1}{|M|} \sum_{m \in M} |r_{jm}|, \qquad w_j = \frac{\bar{r}_j + \epsilon}{\sum_{l \in O}(\bar{r}_l + \epsilon)}$$

The *k* nearest complete rows are retrieved using weighted masked Euclidean distance:

$$d(\mathbf{x}^*, \mathbf{c}_i) = \sqrt{\sum_{j \in O} w_j \left(x^*_j - c_{ij}\right)^2}$$

---

## Installation

### Option A — Conda (recommended)

```bash
git clone https://github.com/yourusername/TabRAG-Imputer.git
cd TabRAG-Imputer
conda env create -f environment.yml
conda activate RAGImputation
```

### Option B — Pip

```bash
git clone https://github.com/yourusername/TabRAG-Imputer.git
cd TabRAG-Imputer
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### API key configuration

TabRAG-Imputer supports multiple LLM providers. Create a `.env` file in the project root and add the key(s) for the provider(s) you intend to use:

```env
# Only the key(s) you need — unused entries can be omitted
API_KEY_GEMINI=your_gemini_key_here
API_KEY_OPEN_ROUTER=your_openrouter_key_here
API_KEY_GPT=your_openai_key_here
API_KEY_CLAUDE=your_anthropic_key_here
```

> **Tip:** Gemini 3.0 Flash (`google/gemini-3-flash-preview`) is the recommended default — it offers competitive imputation accuracy at low cost.

---

## Quick start

### Imputation

```python
import pandas as pd
from algorithms.rag_imputer import RAGImputer

df = pd.read_csv("data/pima-indians-diabetes/pima_diabetes.csv")
X_train, X_test_missing = ...  # your train/test split with NaNs in X_test_missing

imputer = RAGImputer(
    n_neighbors=5,
    llm_model_name="gemini-2.0-flash",
    llm_api="gemini",               # "gemini" | "open_router" | "gpt" | "claude"
    dataset_name="Pima Indians Diabetes",
)

imputer.fit(X_train)
X_imputed = imputer.transform(X_test_missing)
```

### Explainability

After imputation, call `explain()` to get an LLM-generated paragraph for every row
that had missing values. It reuses the same retrieved neighbors, so no extra retrieval
cost is incurred.

```python
explanations = imputer.explain(X_test_missing, X_imputed)

for i, text in enumerate(explanations):
    print(f"[Row {i}] {text}\n")
```

`explain()` accepts an optional `row_indices` list if you only need a subset:

```python
explanations = imputer.explain(X_test_missing, X_imputed, row_indices=[0, 3, 7])
```

See [`examples/quickstart_explain.py`](examples/quickstart_explain.py) for a
self-contained runnable script covering the full workflow — data loading, artificial
missing-value injection, imputation, and explanation output.

---

## Key parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_neighbors` | `int` | `10` | Number of complete rows retrieved as context (*k*). Higher values provide richer context but increase LLM prompt length and cost. |
| `llm_api` | `str` | `"gemini"` | LLM provider to use. Options: `"gemini"`, `"open_router"`, `"gpt"`, `"claude"`. |
| `llm_model_name` | `str` | — | Model identifier string for the chosen provider (e.g. `"google/gemini-3-flash-preview"`). |
| `dataset_name` | `str` | `""` | Human-readable dataset name included in the prompt for context. |

> **Choosing *k*:** Our ablation study shows that *k* = 10 provides the best accuracy–cost trade-off across continuous and mixed datasets. For high-dimensional categorical datasets, *k* = 5 may be preferable to control prompt length and inference time.

---

## Supported missingness mechanisms

TabRAG-Imputer can be evaluated under any mechanism supported by the [`mdatagen`](https://github.com/ArthurMangussi/mdatagen) library:

| Mechanism | Description |
|---|---|
| **MAR** — Missing At Random | Missingness depends on observed features |
| **MNAR** — Missing Not At Random | Missingness depends on the missing values themselves |

MCAR is intentionally excluded from the benchmark as it does not reflect realistic data-generating processes.

---

## Reproducing the paper experiments

The full experimental pipeline — covering all 20 datasets, both mechanisms, three missing rates (5%, 10%, 20%), and all five baselines — can be reproduced by running:

```bash
python run_experiments.py --config configs/full_benchmark.yml
```

Results are saved to `results/` as CSV files and are compatible with the evaluation scripts in `utils/MyResults.py`.

---

## Citation

If you use TabRAG-Imputer in your research, please cite:

```bash
Information will appear as soon as possible
```


---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
