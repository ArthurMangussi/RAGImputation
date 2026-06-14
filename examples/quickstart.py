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