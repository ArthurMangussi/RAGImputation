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

The snippet below runs a complete imputation benchmark on any dataset using
5-fold cross-validation and a MAR missingness mechanism.

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from mdatagen.multivariate.mMAR import mMAR
from utils.MyPreprocessing import PreprocessingDatasets
from utils.MyResults import AnalysisResults
from algorithms.rag_imputer import RAGImputer

# ── 1. Load your dataset ──────────────────────────────────────────────────────
df = pd.read_csv("data/pima-indians-diabetes/pima_diabetes.csv")
X = df.drop(columns="target")
y = df["target"].values

# ── 2. Cross-validation loop ──────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(cv.split(X.values, y), start=1):
    X_train = pd.DataFrame(X.values[train_idx], columns=X.columns)
    X_test  = pd.DataFrame(X.values[test_idx],  columns=X.columns)

    # Normalise — fit on train only to prevent data leakage
    scaler      = PreprocessingDatasets.inicializa_normalizacao(X_train)
    X_train_norm = PreprocessingDatasets.normaliza_dados(scaler, X_train)
    X_test_norm  = PreprocessingDatasets.normaliza_dados(scaler, X_test)

    # ── 3. Inject missing values (MAR, 30%) ───────────────────────────────────
    X_test_missing = (
        mMAR(X=X_test_norm, y=y[test_idx], n_xmiss=X_test_norm.shape[1])
        .random(missing_rate=30)
        .drop(columns="target")
    )

    # ── 4. Fit and impute ─────────────────────────────────────────────────────
    imputer = RAGImputer(
        n_neighbors=10,              # number of retrieved context rows (k)
        llm_api="gemini",            # provider: "gemini" | "open_router" | "gpt" | "claude"
        llm_model_name="google/gemini-3-flash-preview",
        dataset_name="Pima Indians Diabetes",
    )
    imputer.fit(X_train_norm.values)
    X_imputed = imputer.transform(X_test_missing.values)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)
    mae_mean, mae_std = AnalysisResults.gera_resultado_multiva(
        resposta=X_imputed_df,
        dataset_normalizado_md=X_test_missing,
        dataset_normalizado_original=X_test_norm,
    )
    print(f"Fold {fold} — MAE: {mae_mean:.3f} ± {mae_std:.3f}")
```

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
