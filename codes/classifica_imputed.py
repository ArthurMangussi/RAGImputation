# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'
import sys
sys.path.append("./")

import xgboost as xgb
from utils.MeLogSingle import MeLogger
from utils.MyPreprocessing import PreprocessingDatasets
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

import os

from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score

def pipeline_attacked_classification_performance(
                                                 model_impt:str,
                                                 md_mechanism:str,
                                                 names_data:list
                                                 ):
    """
    Main Pipeline to perform the classification task across the baseline datasets 
    """
    _logger = MeLogger()
    diretorio = "./results"
    complete_path = os.path.join(diretorio,model_impt,"DatasetsImputed",md_mechanism)
    
    classification_metrics = {"Dataset":[],
                                        "Missing rate":[],
                                        "F1-score":[],
                                        "Accuracy":[],
                                        "Recall":[],
                                        "AUC":[],
                                        }
    
    for name in names_data:
        for missing_rate in [5,10,20]:
            df_path = complete_path + f"/{name}_{model_impt}_md{missing_rate}.csv"
            dados = pd.read_csv(df_path)
    
            try:                
                df = dados.copy()
                cv = StratifiedKFold()
                model = xgb.XGBClassifier()
                X = df.drop(columns = 'target')
                y = df['target'].values
                x_cv = X.values

                fold = 0
                for train_index, test_index in cv.split(x_cv, y):
                    x_treino, x_teste = x_cv[train_index], x_cv[test_index]
                    y_treino, y_teste = y[train_index], y[test_index]

                    X_treino = pd.DataFrame(x_treino, columns=X.columns)                    
                    X_teste = pd.DataFrame(x_teste, columns=X.columns) 

                    # Inicializando o normalizador (scaler)
                    scaler = PreprocessingDatasets.inicializa_normalizacao(X_treino)

                    # Normalizando os dados
                    X_treino_norm = PreprocessingDatasets.normaliza_dados(scaler, X_treino)
                    X_teste_norm = PreprocessingDatasets.normaliza_dados(scaler, X_teste)

                    model.fit(X_treino_norm, y_treino)

                    y_pred = model.predict(X_teste_norm)

                    if len(np.unique(y)) == 2:
                        _logger.info(f"Dataset {name} is binary -> {np.unique(y)}")
                        f1 = f1_score(y_true=y_teste, y_pred=y_pred)
                        acc = accuracy_score(y_true=y_teste, y_pred=y_pred)
                        rec = recall_score(y_true=y_teste, y_pred=y_pred)
                        auc = roc_auc_score(y_true=y_teste, y_score=model.predict_proba(X_teste_norm)[:,1])
                    else:
                        _logger.info(f"Dataset {name} is Multiclass with {len(np.unique(y))} classes -> {np.unique(y)}")
                        f1 = f1_score(y_true=y_teste, y_pred=y_pred, average="micro")
                        acc = accuracy_score(y_true=y_teste, y_pred=y_pred)
                        rec = recall_score(y_true=y_teste, y_pred=y_pred, average="micro")
                        auc = roc_auc_score(y_true=y_teste, 
                                            y_score=model.predict_proba(X_teste_norm),
                                            multi_class="ovr")

                    classification_metrics["Dataset"].append(name)
                    classification_metrics["Missing rate"].append(missing_rate)
                    classification_metrics["F1-score"].append(round(f1,3))
                    classification_metrics["Accuracy"].append(round(acc,3))
                    classification_metrics["Recall"].append(round(rec,3))
                    classification_metrics["AUC"].append(round(auc,3))
                    fold += 1
            
            except Exception as erro:
                _logger.debug(f"Erro: {erro}")
        
    resultados = pd.DataFrame(classification_metrics)
    resultados.to_csv(f"./results/{model_impt}/Resultados/classification_{model_impt}_{mecanismo}.csv", index=False)
    return _logger.info("Resultados Classificação salvos com sucesso!")
    

if __name__ == "__main__":

    names_dataset  = ["wiscosin",
            "hepatitis",
            "mathernal_risk",
            "chronic",
            "stalog",
            "student_port",
            "german-credit"]

    for mecanismo in ["MAR", "MNAR"]:
    
        pipeline_attacked_classification_performance(
                                                    model_impt="gain",
                                                    md_mechanism=mecanismo,
                                                    names_data=names_dataset)
        pipeline_attacked_classification_performance(
                                                    model_impt="knn",
                                                    md_mechanism=mecanismo,
                                                    names_data=names_dataset)
        pipeline_attacked_classification_performance(
                                                    model_impt="mice",
                                                    md_mechanism=mecanismo,
                                                    names_data=names_dataset)
        pipeline_attacked_classification_performance(
                                                    model_impt="missForest",
                                                    md_mechanism=mecanismo,
                                                    names_data=names_dataset)
        pipeline_attacked_classification_performance(
                                                    model_impt="ragGemini",
                                                    md_mechanism=mecanismo,
                                                    names_data=names_dataset)
                                                    