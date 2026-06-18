import os 
import pandas as pd

for name_dataset in ["wiscosin",
            "hepatitis",
            "mathernal_risk",
            "chronic",
            "stalog",
            "student_port",
            "german-credit"
            "continuous-variation-1",
            "continuous-variation-2",
            "continuous-variation-3",
            "continuous-variation-4",
            "continuous-variation-5",
            "continuous-variation-6",
            "categorical-variation-1",
            "categorical-variation-2",
            "categorical-variation-3",
            "categorical-variation-4",
            "categorical-variation-5",
            "mixed-variation-1",
            "mixed-variation-2",
            "mixed-variation-3",
            "mixed-variation-4",
            ]:

    for model_impt in ["missForest",
                    "knn",
                    "mice",
                    "gain",
                    "ragGemini"
                      ]:
        
        for mr in [5,10,20]:

            folds = []
            for fold in range (5):
              arq = f"{name_dataset}_{model_impt}_fold{fold}_md{mr}.csv"
              path = f"./results/{model_impt}/Datasets/MAR_Multivariado"
              df = pd.read_csv(os.path.join(path,arq))
              folds.append(df)
                  
            df_unificado = pd.concat(folds, ignore_index=True)
            df_unificado.to_csv(f"./results/{model_impt}/{name_dataset}_{model_impt}_md{mr}.csv", index=False)        

print("done")