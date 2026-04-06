# RAGImputation

A novel Retrieval-Augmented Generation (RAG) approach for missing data imputation. Instead of relying purely on statistical distance (like KNN) or generative capacity (like pure LLMs), RAGImputer combines both: it retrieves semantically similar complete rows via a **correlation-weighted masked Euclidean distance**, then feeds them as context to an LLM that produces the final imputed values.

## Method

The core retrieval stage (`_retrieve`) uses a **two-stage** approach:

**Stage 1 — Correlation-based feature weighting.** For a query row $\mathbf{x}^*$ with observed feature indices $O$ and missing feature indices $M$, compute a weight for each observed feature based on how strongly it correlates with the missing features:

$$\bar{r}_j = \frac{1}{|M|} \sum_{m \in M} |r_{jm}|, \quad j \in O$$

where $r_{jm}$ is the Pearson correlation between features $j$ and $m$ estimated from the complete training rows. Weights are then smoothed and normalised:

$$w_j = \frac{\bar{r}_j + \epsilon}{\displaystyle\sum_{l \in O} (\bar{r}_l + \epsilon)}, \quad \epsilon = 0.05$$

**Stage 2 — Weighted masked Euclidean distance.** Using only the observed features and the weights above, compute the distance between the query and each complete context row $\mathbf{c}_i$:

$$d(\mathbf{x}^*, \mathbf{c}_i) = \sqrt{\sum_{j \in O} w_j \left(x^*_j - c_{ij}\right)^2}$$

The $k$ nearest context rows are selected and serialised as natural-language text, then injected into an LLM prompt that produces the imputed values for the missing features.

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate RAGImputation
```

### Pip

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Unix: source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

If using LLM-based imputation, configure your API keys in a `.env` file at the project root:

```env
API_KEY_OPEN_ROUTER=...
API_KEY_GEMINI=...
API_KEY_GPT=...
API_KEY_CLAUDE=...
```

## Use Case

Below is a minimal example reproducing the experimental benchmark used to validate the RAG imputer under the **MAR** (Missing At Random) mechanism with 5-fold cross-validation:

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from mdatagen.multivariate.mMAR import mMAR
from utils.MyPreprocessing import PreprocessingDatasets
from utils.MyResults import AnalysisResults
from utils.MyUtils import MyPipeline
from algorithms.rag_imputer import RAGImputer

if __name__ == "__main__":
    # 1. Load dataset
    df = pd.read_csv("./data/pima-indians-diabetes/pima_diabetes.csv")
    X, y = df.drop(columns="target"), df["target"].values

    # 2. Cross-validation loop
    cv = StratifiedKFold(n_splits=5)
    for train_idx, test_idx in cv.split(X.values, y):
        X_train = pd.DataFrame(X.values[train_idx], columns=X.columns)
        X_test = pd.DataFrame(X.values[test_idx], columns=X.columns)

        # Normalise
        scaler = PreprocessingDatasets.inicializa_normalizacao(X_train)
        X_train_norm = PreprocessingDatasets.normaliza_dados(scaler, X_train)
        X_test_norm = PreprocessingDatasets.normaliza_dados(scaler, X_test)

        # Inject missing values (MAR, 30%)
        impt_md_test = mMAR(
            X=X_test_norm,
            y=y[test_idx],
            n_xmiss=X_test_norm.shape[1],
        )
        X_teste_norm_md = impt_md_test.random(missing_rate=30)
        X_teste_norm_md = X_teste_norm_md.drop(columns="target")

        # 3. Fit RAGImputer on complete training data, impute test set
        model = RAGImputer(
            n_neighbors=10,
            llm_api="open_router",
            llm_model_name="google/gemini-3-flash-preview",
            dataset_name="Pima Indians Diabetes",
        )
        model.fit(X_train_norm.values)
        X_imputed = model.transform(X_teste_norm_md.values)

        df_output_md_teste = pd.DataFrame(X_imputed, columns=X.columns)
        # 4. Evaluate
        mae_mean, mae_std = AnalysisResults.gera_resultado_multiva(
            resposta=df_output_md_teste,
            dataset_normalizado_md=X_teste_norm_md,
            dataset_normalizado_original=X_test_norm,
        )
        print(f"MAE: {mae_mean:.3f} ± {mae_std:.3f}")

```

## Author

**Arthur Dantas Mangussi**
Aeronautics Institute of Technologies (ITA) - Brazil
University of Coimbra (UC) - Portugal
`mangussiarthur@gmail.com`
