import sys

sys.path.append("./")
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from utils.MyMain import BenchmarkPipeline
from utils.MyUtils import MyPipeline
from utils.MeLogSingle import MeLogger
from utils.MyResults import AnalysisResults
from time import sleep
from mdatagen.multivariate.mMNAR import mMNAR
from time import perf_counter
import os

from algorithms.llm import DATASET_NAMES, llm_impute, MAPPED_LLMS
from algorithms.rag_imputer import RAGImputer

# Register RAG in the mapped-LLM table so directory names are consistent
MAPPED_LLMS = {**MAPPED_LLMS, "rag-aggregation": "ragAgg", "rag-llm": "ragLLM"}


def pipeline_benchmark_imputation(
    model_impt: str, mecanismo: str, tabela_resultados: dict, api:str="open_router"
):
    "Main pipeline to perform imputation MCAR multivariate mechanism."
    _logger = MeLogger()

    # Cria diretórios para salvar os resultados do experimento
    os.makedirs(
        f"./results/{MAPPED_LLMS[model_impt]}/Tempos/{mecanismo}_Multivariado", exist_ok=True
    )
    os.makedirs(
        f"./results/{MAPPED_LLMS[model_impt]}/Datasets/{mecanismo}_Multivariado", exist_ok=True
    )
    os.makedirs(
        f"./results/{MAPPED_LLMS[model_impt]}/Resultados/{mecanismo}_Multivariado", exist_ok=True
    )

    for md in tabela_resultados["missing_rate"]:
        for dados, nome in zip(
            tabela_resultados["datasets"], tabela_resultados["nome_datasets"]
        ):
            df = dados.copy()
            X = df.drop(columns="target")
            y = df["target"].values
            binary_features = MyPipeline.get_binary_features(data=df)
            imputation_time = {}

            _logger.info(f"Dataset = {nome} com MD = {md} no {model_impt}\n")

            fold = 0
            cv = StratifiedKFold(n_splits=5)
            x_cv = X.values

            
            for train_index, test_index in cv.split(x_cv, y):
                _logger.info(f"Fold = {fold}")
                x_treino, x_teste = x_cv[train_index], x_cv[test_index]
                y_treino, y_teste = y[train_index], y[test_index]

                X_treino = pd.DataFrame(x_treino, columns=X.columns)
                X_teste = pd.DataFrame(x_teste, columns=X.columns)

                # Geração dos missing values em cada conjunto de forma independente
                impt_md_train = mMNAR(
                    X=X_treino,
                    y=y_treino,
                    n_xmiss=X_treino.shape[1],
                    threshold=1,
                )
                X_treino_md = impt_md_train.random(missing_rate=md)
                X_treino_md = X_treino_md.drop(columns="target")

                impt_md_test = mMNAR(
                    X=X_teste,
                    y=y_teste,
                    n_xmiss=X_teste.shape[1],
                    threshold=1,
                )
                X_teste_md = impt_md_test.random(missing_rate=md)
                X_teste_md = X_teste_md.drop(columns="target")

                inicio_imputation = perf_counter()
                attempt = 0
                max_attempts = 3
                while attempt < max_attempts:
                    try:
                        if model_impt in ("rag-aggregation", "rag-llm"):
                            rag_mode = "aggregation" if model_impt == "rag-aggregation" else "llm"
                            imputer = RAGImputer(
                                n_neighbors=5,
                                mode=rag_mode,
                                llm_model_name="openai/gpt-4.1-nano",
                                llm_api=api,
                                dataset_name=DATASET_NAMES.get(nome, nome),
                            )
                            imputer.fit(X_treino)
                            imputed_arr = imputer.transform(X_teste_md)
                            df_output_md_teste = pd.DataFrame(
                                imputed_arr, columns=X_teste_md.columns
                            )
                        else:
                            df_output_md_teste = llm_impute(
                                dataset_name=DATASET_NAMES[nome],
                                X_teste_norm_md=X_teste_md,
                                model_name=model_impt,
                                api=api
                            )
                        break
                    except Exception as e:
                        attempt += 1
                        if attempt == max_attempts:
                            _logger.info("Max retries reached. Service still unavailable.")
                            raise
                        wait_time = 2 ** attempt
                        print(f"Model overloaded. Retrying in {wait_time}s... (Attempt {attempt}/{max_attempts})")
                        sleep(wait_time)

                fim_imputation = perf_counter()


                imputation_time[f"{model_impt}_mr{md}_fold{fold}_{nome}"] = round(
                    fim_imputation - inicio_imputation, 3
                )

                # Encode das variávies binárias
                output_md_test = MyPipeline.encode_features_categoricas(
                    list_binary_features=binary_features,
                    imputed_dataset=df_output_md_teste,
                )

                # Calculando MAE para a imputação no conjunto de teste
                (
                    mae_teste_mean,
                    mae_teste_std,
                ) = AnalysisResults.gera_resultado_multiva_nrmse(
                    resposta=output_md_test,
                    dataset_normalizado_md=X_teste_md,
                    dataset_normalizado_original=X_teste,
                )

                tabela_resultados[f"{MAPPED_LLMS[model_impt]}/{nome}/{md}/{fold}/MAE"] = {
                    "teste": round(mae_teste_mean, 3)
                }

                # Dataset imputado
                data_imputed = pd.DataFrame(output_md_test.copy(), columns=X.columns)
                data_imputed["target"] = y_teste
                
                data_imputed.to_csv(
                    f"./results/{MAPPED_LLMS[model_impt]}/Datasets/{mecanismo}_Multivariado/{nome}_{MAPPED_LLMS[model_impt]}_fold{fold}_md{md}.csv",
                    index=False,
                )
                fold += 1

            resultados_final = AnalysisResults.extrai_resultados(tabela_resultados)

            # Resultados da imputação: salvos por dataset para ganhar tempo de processamento
            resultados_mecanismo = (
                AnalysisResults.calcula_metricas_estatisticas_resultados(
                    resultados_final, 1, fold
                )
            )

            resultados_mecanismo.to_csv(
                f"./results/{MAPPED_LLMS[model_impt]}/Resultados/{mecanismo}_Multivariado/{nome}_{MAPPED_LLMS[model_impt]}_{mecanismo}.csv",
            )
            pd.DataFrame({"Tempos": imputation_time}).to_csv(
                f"./results/{MAPPED_LLMS[model_impt]}/Tempos/{mecanismo}_Multivariado/{nome}_{MAPPED_LLMS[model_impt]}_{mecanismo}.csv"
            )

    return _logger.info(f"Imputation_{model_impt}_done!")


if __name__ == "__main__":

    diretorio = "./data"
    datasets = MyPipeline.carrega_datasets(diretorio)

    pipeline = BenchmarkPipeline(datasets)
    tabela_resultados = pipeline.cria_tabela()

    mecanismo = "MNAR"
        
    pipeline_benchmark_imputation(
        "anthropic/claude-sonnet-4.5", 
        mecanismo, 
        tabela_resultados)