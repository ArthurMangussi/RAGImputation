# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'

from utilsMsc.MyADML import AdversarialML
from utilsMsc.MyComplexity import ComplexityDatasets
import pandas as pd
from utilsMsc.MyUtils import MyPipeline
from utilsMsc.MyResults import AnalysisResults

if __name__ == '__main__':

    diretorio = "./data"
    datasets = MyPipeline.carrega_datasets(diretorio)

    adv_ml = AdversarialML(datasets)
    tabela_resultados = adv_ml.cria_tabela()

    mecanismo = "Baseline"

    # ComplexityDatasets.cria_arquivo_arff(mecanismo,tabela_resultados)

    bs = {}
    for nome in tabela_resultados['nome_datasets']:
        print(f"Complexidade --> {nome}")
        path = f'./Complexidade/{mecanismo}_Multivariado/Arquivos/{nome}.arff'
        bs[nome] = ComplexityDatasets.analisa_complexidade(path)

    pd.DataFrame(bs).to_excel(
        f'./Complexidade/{mecanismo}_Multivariado/baseline.xlsx'
    )
