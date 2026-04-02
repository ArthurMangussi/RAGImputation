# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'
import os
import sys

sys.path.append("./")

from utils.MyComplexity import ComplexityDatasets
import pandas as pd
from utils.MyMain import BenchmarkPipeline
from utils.MyUtils import MyPipeline

if __name__ == '__main__':

    diretorio = "./data"
    datasets = MyPipeline.carrega_datasets(diretorio)

    pipeline = BenchmarkPipeline(datasets)
    tabela_resultados = pipeline.cria_tabela_sintetico()

    #for nome in os.listdir("./data/synthetic"): 
    #    ComplexityDatasets.cria_arquivo_arff(f"/home/gpu-10-2025/Área de trabalho/RAGImputation/data/synthetic/{nome}", nome.split(".")[0])

    bs = {}
    for nome in os.listdir("./data/synthetic"): 
        print(f"Complexidade --> {nome}")
        path = f'./Complexidade/{nome.split(".")[0]}.arff'
        bs[nome] = ComplexityDatasets.analisa_complexidade(path)

    pd.DataFrame(bs).to_excel(
        f'./Complexidade/baseline.xlsx'
    )
