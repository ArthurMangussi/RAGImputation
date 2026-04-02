# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = "Arthur Dantas Mangussi"

import pandas as pd
from utils.pycol import Complexity

import arff
import os


class ComplexityDatasets:
    def __init__(self, baseline_dataset) -> None:
        self.base = baseline_dataset

    # ------------------------------------------------------------------------
    @staticmethod
    def analisa_complexidade(path: str) -> dict:
        complexity = Complexity(path)
        return {
            # Feature Overlap
            "f1v": complexity.F1v(),
            "f2": complexity.F2(),
            # Instance Overlap
            "n3": complexity.N3(),
            "n4": complexity.N4(),
            # Structural Overlap
            "n1": complexity.N1(),
            "n2": complexity.N2(),
        }

    # ------------------------------------------------------------------------
    @staticmethod
    def cria_arquivo_arff(
        path: str,
        type_attack: str,
        mechanism: str,
        model_impt: str,
        nome_dataset: str,
        mr: int,
    ):

        dados = pd.read_csv(path)

        arff_content = ComplexityDatasets.save_arff(
            data_complex=dados,
            type_attack=type_attack,
            mechanism=mechanism,
            model_impt=model_impt,
            nome_dataset=nome_dataset,
        )

        # Salvar o conteúdo ARFF em um arquivo
        with open(
            f"./Complexidade/{type_attack}/{mechanism}/{model_impt}/{nome_dataset}_md{mr}.arff",
            "w",
        ) as fcom:
            fcom.write(arff_content)

    # ------------------------------------------------------------------------
    @staticmethod
    def save_arff(
        data_complex: pd.DataFrame,
        type_attack: str,
        mechanism: str,
        model_impt: str,
        nome_dataset: str,
    ):

        os.makedirs(
            f"./Complexidade/{type_attack}/{mechanism}/{model_impt}/",
            exist_ok=True,
        )
        atts = ComplexityDatasets.formata_arff(data_complex, nome_dataset)

        dictarff = {
            "attributes": atts,
            "data": data_complex.values.tolist(),
            "relation": f"{nome_dataset}",
        }

        # Criar o arquivo ARFF
        return arff.dumps(dictarff)

    # ------------------------------------------------------------------------
    @staticmethod
    def formata_arff(data_imputed_complete, name):
        attributes = []
        for j in data_imputed_complete:
            if (
                data_imputed_complete[j].dtypes in ["int64", "float64", "float32"]
                and j != "target"
            ):
                attributes.append((j, "NUMERIC"))
            elif j == "target":
                if (
                    name == "hcv_egyptian"
                    or name == "npha"
                    or name == "vertebral_column"
                    or name == "mathernal-risk"
                    or name == "iris"
                    or name == "wine"
                    or name == "contraceptive-methods"
                ):
                    attributes.append(
                        (j, sorted(data_imputed_complete[j].unique().astype(str)))
                    )
                else:
                    attributes.append((j, ["1.0", "0.0"]))
            else:
                attributes.append(
                    (j, sorted(data_imputed_complete[j].unique().astype(str)))
                )

        return attributes
