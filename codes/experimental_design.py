import sys

sys.path.append("./")
import multiprocessing
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from utils.MyMain import BenchmarkPipeline
from utils.MyModels import ModelsImputation
from utils.MyUtils import MyPipeline
from utils.MyPreprocessing import PreprocessingDatasets
from utils.MeLogSingle import MeLogger
from utils.MyResults import AnalysisResults


from mdatagen.multivariate.mMNAR import mMNAR

from time import perf_counter
import os


def pipeline_benchmark_imputation(
    model_impt: str, mecanismo: str, tabela_resultados: dict
):
    "Main pipeline to perform imputation MCAR multivariate mechanism."
    _logger = MeLogger()

    # Cria diretórios para salvar os resultados do experimento
    os.makedirs(
        f"./results/{model_impt}/Tempos/{mecanismo}_Multivariado", exist_ok=True
    )
    os.makedirs(
        f"./results/{model_impt}/Datasets/{mecanismo}_Multivariado", exist_ok=True
    )
    os.makedirs(
        f"./results/{model_impt}/Resultados/{mecanismo}_Multivariado", exist_ok=True
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
            cv = StratifiedKFold()
            x_cv = X.values

            for train_index, test_index in cv.split(x_cv, y):
                _logger.info(f"Fold = {fold}")
                x_treino, x_teste = x_cv[train_index], x_cv[test_index]
                y_treino, y_teste = y[train_index], y[test_index]

                X_treino = pd.DataFrame(x_treino, columns=X.columns)
                X_teste = pd.DataFrame(x_teste, columns=X.columns)

                # Inicializando o normalizador (scaler)
                scaler = PreprocessingDatasets.inicializa_normalizacao(X_treino)

                # Normalizando os dados
                X_treino_norm = PreprocessingDatasets.normaliza_dados(scaler, X_treino)
                X_teste_norm = PreprocessingDatasets.normaliza_dados(scaler, X_teste)

                # Geração dos missing values em cada conjunto de forma independente
                impt_md_train = mMNAR(
                    X=X_treino_norm,
                    y=y_treino,
                    threshold=1,                    
                    n_xmiss=X_treino_norm.shape[1]              
                   
                )
                X_treino_norm_md = impt_md_train.random(missing_rate=md)
                X_treino_norm_md = X_treino_norm_md.drop(columns="target")
                

                impt_md_test = mMNAR(
                    X=X_teste_norm,
                    y=y_teste,
                    n_xmiss=X_teste_norm.shape[1],
                    threshold=1
                    
                )
                X_teste_norm_md = impt_md_test.random(missing_rate=md)
                X_teste_norm_md = X_teste_norm_md.drop(columns="target")
                
                        
                inicio_imputation = perf_counter()

                # Inicializando e treinando o modelo
                model_selected = ModelsImputation()

                model = model_selected.choose_model(
                    model=model_impt,
                    x_train=X_treino_norm_md,
                    x_test=X_teste_norm_md,
                    x_test_complete=X_teste_norm,
                    binary_val=binary_features,
                    x_train_complete=X_treino_norm,
                    input_shape=X_treino_norm.shape[1],
                )

                fim_imputation = perf_counter()

                imputation_time[f"{model_impt}_mr{md}_fold{fold}_{nome}"] = round(
                    fim_imputation - inicio_imputation, 3
                )

                # Imputação dos missing values nos conjuntos de treino e teste
                try:
                    if model_impt == "tabpfn":
                        output_md_test = model.impute(
                            X_teste_norm_md,
                            n_permutations=5,  # Fewer permutations for a quicker example
                        )
                    else:
                        output_md_test = model.transform(
                            X_teste_norm_md.iloc[:, :].values
                        )

                except AttributeError:
                    fatores_latentes_test = model.fit(X_teste_norm_md.iloc[:, :].values)
                    output_md_test = model.predict(X_teste_norm_md.iloc[:, :].values)

                # Encode das variávies binárias
                df_output_md_teste = pd.DataFrame(output_md_test, columns=X.columns)
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
                    dataset_normalizado_md=X_teste_norm_md,
                    dataset_normalizado_original=X_teste_norm,
                )

                tabela_resultados[f"{model_impt}/{nome}/{md}/{fold}/MAE"] = {
                    "teste": round(mae_teste_mean, 3)
                }

                # Dataset imputado
                data_imputed = pd.DataFrame(output_md_test.copy(), columns=X.columns)
                data_imputed["target"] = y_teste

                data_imputed.to_csv(
                    f"./results/{model_impt}/Datasets/{mecanismo}_Multivariado/{nome}_{model_impt}_fold{fold}_md{md}.csv",
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
                f"./results/{model_impt}/Resultados/{mecanismo}_Multivariado/{nome}_{model_impt}_{mecanismo}.csv",
            )
            pd.DataFrame({"Tempos": imputation_time}).to_csv(
                f"./results/{model_impt}/Tempos/{mecanismo}_Multivariado/{nome}_{model_impt}_{mecanismo}.csv"
            )

    return _logger.info(f"Imputation_{model_impt}_done!")


if __name__ == "__main__":

    diretorio = "./data"
    datasets = MyPipeline.carrega_datasets(diretorio)

    pipeline = BenchmarkPipeline(datasets)
    tabela_resultados = pipeline.cria_tabela()

    mecanismo = "MNAR"

    pipeline_benchmark_imputation("softImpute", mecanismo, tabela_resultados)
    pipeline_benchmark_imputation("knn", mecanismo, tabela_resultados)
    pipeline_benchmark_imputation("missForest", mecanismo, tabela_resultados)
    pipeline_benchmark_imputation("mice", mecanismo, tabela_resultados)    
    pipeline_benchmark_imputation("saei", mecanismo, tabela_resultados)
    pipeline_benchmark_imputation("tabpfn", mecanismo, tabela_resultados)
