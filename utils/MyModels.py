# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = "Arthur Dantas Mangussi"

# Partial Multiple Imputation with Variational Autoencoders
from algorithms.pmivae import PMIVAE
from algorithms.vae_pmivae import ConfigVAE

# Siamese Autoencoder
from algorithms.saei import ConfigSAE, SAEImp, DataSets

# MICE, KNN, Dumb, missForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer

# Soft impute
from algorithms.soft_impute import SoftImpute

# Generative Adversarial Imputation Networks
from algorithms.gain import Gain

# RAG Imputer
from algorithms.rag_imputer import RAGImputer

# LLM
from algorithms.llm import MAPPED_LLMS, LLMWrapper

# TabPFN
from tabpfn import TabPFNClassifier, TabPFNRegressor

# from tabpfn_extensions import unsupervised

import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import mean_squared_error

from utils.MeLogSingle import MeLogger
from utils.MyUtils import MyPipeline

# Ignorar todos os avisos
warnings.filterwarnings("ignore")


class ModelsImputation:
    def __init__(self):
        self._logger = MeLogger()

    # ------------------------------------------------------------------------
    @staticmethod
    def model_mice(dataset_train: pd.DataFrame):
        imputer = IterativeImputer(max_iter=1000)
        mice = imputer.fit(dataset_train.iloc[:, :].values)

        return mice

    # ------------------------------------------------------------------------
    @staticmethod
    def model_knn(dataset_train: pd.DataFrame):
        imputer = KNNImputer(n_neighbors=5)
        knn = imputer.fit(dataset_train.iloc[:, :].values)

        return knn

    # ------------------------------------------------------------------------
    @staticmethod
    def model_dumb(dataset_train: pd.DataFrame, binary_vals: list[str]):

        numeric_imputer = SimpleImputer(strategy="mean")
        categorical_imputer = SimpleImputer(strategy="most_frequent")

        num_vals = [col for col in dataset_train.columns if col not in binary_vals]
        try:
            copy_binary_vals = binary_vals.copy()
            copy_binary_vals.remove("target")
        except ValueError:
            print(binary_vals)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_imputer, num_vals),
                ("cat", categorical_imputer, copy_binary_vals),
            ]
        )

        dumb = preprocessor.fit(dataset_train)
        return dumb

    # ------------------------------------------------------------------------
    @staticmethod
    def model_autoencoder_pmivae(dataset_train: pd.DataFrame, params: dict):
        original_shape = dataset_train.shape
        vae_config = ConfigVAE()
        vae_config.verbose = 0
        vae_config.batch_size = 128
        vae_config.validation_split = 0.2
        vae_config.input_shape = (original_shape[1],)
        vae_config.epochs = params["n_epochs"]
        vae_config.latent_dimension = params["latent_dimension"]
        vae_config.neurons = params["neurons"]
        vae_config.dropout_fc = [0.3] * len(params["neurons"])

        pmivae_model = PMIVAE(vae_config, num_samples=200)
        model = pmivae_model.fit(dataset_train)

        return model

    # ------------------------------------------------------------------------
    @staticmethod
    def model_softimpute(dataset_train: pd.DataFrame):
        imputer = SoftImpute()
        soft_impute = imputer.fit(dataset_train.iloc[:, :].values)
        return soft_impute

    # ------------------------------------------------------------------------
    @staticmethod
    def model_gain(dataset_train: pd.DataFrame):
        imputer = Gain(batch_size=32, iterations=10000)
        gain = imputer.fit(dataset_train.iloc[:, :].values)
        return gain

    # ------------------------------------------------------------------------
    @staticmethod
    def model_missForest(dataset_train: pd.DataFrame):
        rf = RandomForestRegressor(
            n_jobs=-1, criterion="absolute_error", n_estimators=10, random_state=42
        )
        imputer = IterativeImputer(estimator=rf)
        missForest = imputer.fit(dataset_train.iloc[:, :].values)

        return missForest

    # ------------------------------------------------------------------------
    @staticmethod
    def model_tabpfn(dataset_train: pd.DataFrame):
        clf = TabPFNClassifier(n_estimators=3)
        reg = TabPFNRegressor(n_estimators=3)

        # Initialize the main unsupervised model
        tabpfn = unsupervised.TabPFNUnsupervisedModel(
            tabpfn_clf=clf,
            tabpfn_reg=reg,
        )
        tabpfn.fit(dataset_train)

        return tabpfn

    # ------------------------------------------------------------------------
    @staticmethod
    def set_vae_config(X, neurons, latent_dimension, n_epochs):
        original_shape = X.shape
        vae_config = ConfigVAE()
        vae_config.verbose = 0
        vae_config.batch_size = 128
        vae_config.validation_split = 0.2
        vae_config.input_shape = (original_shape[1],)
        vae_config.neurons = neurons
        vae_config.latent_dimension = latent_dimension
        vae_config.n_epochs = n_epochs
        vae_config.dropout_fc = [0.3] * len(neurons)

        return vae_config

    # ------------------------------------------------------------------------
    @staticmethod
    def train_pmivae(config, X_train, X_test, X_test_complete):
        pmivae_model = PMIVAE(config, num_samples=200)
        pmivae_model.fit(X_train.iloc[:, :].values)
        output_test = pmivae_model.transform(X_test.iloc[:, :].values)
        mse = mean_squared_error(y_pred=output_test, y_true=X_test_complete)
        return mse

    # ------------------------------------------------------------------------
    @staticmethod
    def GridSearchPMIVAE(X_train, X_test, X_test_complete, param_grid):
        best_score = np.inf
        best_params = {}

        for n_epochs in param_grid["epochs"]:
            for n_latent_dimension in param_grid["latent_dimension"]:
                for nro_layers in [1, 2]:
                    if nro_layers == 1:
                        for nro_neurons in param_grid["neurons"]:
                            neurons = [nro_neurons]
                            vae_config = ModelsImputation.set_vae_config(
                                X_train, neurons, n_latent_dimension, n_epochs
                            )
                            mse = ModelsImputation.train_pmivae(
                                vae_config, X_train, X_test, X_test_complete
                            )
                            if mse < best_score:
                                best_score = mse
                                best_params = {
                                    "n_epochs": n_epochs,
                                    "latent_dimension": n_latent_dimension,
                                    "neurons": neurons,
                                }
                    elif nro_layers == 2:
                        for nro_neurons1 in param_grid["neurons"]:
                            for nro_neurons2 in param_grid["neurons"]:
                                neurons = [nro_neurons1, nro_neurons2]
                                vae_config = ModelsImputation.set_vae_config(
                                    X_train, neurons, n_latent_dimension, n_epochs
                                )
                                mse = ModelsImputation.train_pmivae(
                                    vae_config, X_train, X_test, X_test_complete
                                )
                                if mse < best_score:
                                    best_score = mse
                                    best_params = {
                                        "n_epochs": n_epochs,
                                        "latent_dimension": n_latent_dimension,
                                        "neurons": neurons,
                                    }
                    else:  # nro_layers == 3
                        for nro_neurons1 in param_grid["neurons"]:
                            for nro_neurons2 in param_grid["neurons"]:
                                for nro_neurons3 in param_grid["neurons"]:
                                    neurons = [nro_neurons1, nro_neurons2, nro_neurons3]
                                    vae_config = ModelsImputation.set_vae_config(
                                        X_train, neurons, n_latent_dimension, n_epochs
                                    )
                                    mse = ModelsImputation.train_pmivae(
                                        vae_config, X_train, X_test, X_test_complete
                                    )
                                    if mse < best_score:
                                        best_score = mse
                                        best_params = {
                                            "n_epochs": n_epochs,
                                            "latent_dimension": n_latent_dimension,
                                            "neurons": neurons,
                                        }
        return best_params, best_score

    # ------------------------------------------------------------------------
    @staticmethod
    def modelo_saei(
        dataset_train: pd.DataFrame,
        dataset_test: pd.DataFrame,
        dataset_train_md: pd.DataFrame,
        dataset_test_md: pd.DataFrame,
        input_shape,
    ):
        vae_config = ConfigSAE()
        vae_config.verbose = 0
        vae_config.epochs = 200
        vae_config.input_shape = (input_shape,)

        saei_model = SAEImp()
        prep = MyPipeline()
        x_train_pre = prep.pre_imputed_dataset(dataset_train_md)
        x_test_pre = prep.pre_imputed_dataset(dataset_test_md)

        dados = DataSets(
            x_train=dataset_train,
            x_val=dataset_test,
            x_train_md=dataset_train_md,
            x_val_md=dataset_test_md,
            x_train_pre=x_train_pre,
            x_val_pre=x_test_pre,
        )

        model = saei_model.fit(dados, vae_config)
        return model

    # ------------------------------------------------------------------------
    def choose_model(self, model: str, x_train, **kwargs):
        match model:
            case "tabpfn":
                self._logger.info("[TabPFN] Training...")
                return ModelsImputation.model_tabpfn(x_train)

            case "mice":
                self._logger.info("[MICE] Training...")
                return ModelsImputation.model_mice(x_train)

            case "knn":
                self._logger.info("[KNN] Training...")
                return ModelsImputation.model_knn(x_train)

            case "pmivae":
                self._logger.info("[PMIVAE] GridSearch...")
                params = {
                    "epochs": [200],
                    "latent_dimension": [5, 10],
                    "neurons": [
                        [np.shape(x_train)[0] / 2],
                        [np.shape(x_train)[0] / 2, np.shape(x_train)[0] / 4],
                    ],
                }
                best_params, best_score = ModelsImputation.GridSearchPMIVAE(
                    X_train=x_train,
                    X_test=kwargs["x_test"],
                    param_grid=params,
                    X_test_complete=kwargs["x_test_complete"],
                )

                self._logger.info(f"Best params for PMIVAE: {best_params}")
                self._logger.info(f"Best score found in GridSearch (MSE): {best_score}")

                return ModelsImputation.model_autoencoder_pmivae(
                    x_train.loc[:, :].values, params=best_params
                )

            case "saei":
                self._logger.info("[SAEI] Training...")
                return ModelsImputation.modelo_saei(
                    dataset_train=kwargs["x_train_complete"],
                    dataset_test=kwargs["x_test_complete"],
                    dataset_train_md=x_train,
                    dataset_test_md=kwargs["x_test"],
                    input_shape=kwargs["input_shape"],
                )

            case "mean":
                self._logger.info("[MEAN] Training...")
                return ModelsImputation.model_dumb(x_train, kwargs["binary_val"])

            case "softImpute":
                self._logger.info("[SoftImpute] Training...")
                return ModelsImputation.model_softimpute(x_train)

            case "gain":
                self._logger.info("[GAIN] Training...")
                return ModelsImputation.model_gain(x_train)

            case "missForest":
                self._logger.info("[missForest] Training...")
                return ModelsImputation.model_missForest(x_train)

            case "ragGemini":
                self._logger.info("[RAGImputer] Training...")
                imputer = RAGImputer(
                    n_neighbors=kwargs.get("n_neighbors", 3),
                    feature_weighting=kwargs.get("feature_weighting", "correlation"),
                    llm_model_name=kwargs.get("llm_model_name", "gemini-3-flash-preview"),
                    llm_api=kwargs.get("llm_api", "gemini"),
                    dataset_name=kwargs.get("dataset_name", "Unknown Dataset"),
                    llm_batch_size=kwargs.get("llm_batch_size", 1),
                )
                imputer.fit(x_train)
                return imputer

            case _ if model in MAPPED_LLMS:
                self._logger.info(f"[{model}] Training (LLMWrapper)...")
                feature_names = kwargs.get("feature_names", None)
                if feature_names is None:
                    # fallback if feature_names not provided, try to extract from x_test
                    x_t = kwargs.get("x_test")
                    if hasattr(x_t, "columns"):
                        feature_names = list(x_t.columns)
                    else:
                        feature_names = [f"x{i}" for i in range(x_train.shape[1])]

                wrapper = LLMWrapper(
                    model_name=model,
                    api=kwargs.get("api", "open_router"),
                    dataset_name=kwargs.get("dataset_name", "Unknown Dataset"),
                    feature_names=feature_names,
                )
                wrapper.fit(x_train)
                return wrapper
