from utils.MyPreprocessing import PreprocessingDatasets
from utils.MeLogSingle import MeLogger

import numpy as np
import pandas as pd


class BenchmarkPipeline:
    def __init__(self, datasets: dict):
        self._logger = MeLogger()
        self._prep = PreprocessingDatasets()
        self.datasets = datasets
        self.pima = self.pre_processing_pima()
        self.cleveland = self.pre_processing_cleveland()
        self.wiscosin = self.pre_processing_wiscosin()
        self.parkinsons = self.pre_processing_parkinsons()
        self.hepatitis = self.pre_processing_hepatitis()
        self.mathernal_risk = self.pre_processing_mathernal_rick()
        self.chronic = self.pre_processing_chronic()
        self.stalog = self.pre_processing_stalog_heart()
        # ---------- Novos --------------------------
        self.covid = self.pre_processing_covid()
        self.stroke = self.pre_processing_stroke()
        self.cervical = self.pre_processing_cervical()
        self.wine = self.pre_processing_wine()
        self.german_credit = self.pre_processing_german_credit()
        self.adult = self.pre_processing_adult()
        self.student_math = self.pre_processing_student_math()
        self.student_port = self.pre_processing_student_port()
        self.compass_4k = self.pre_processing_compass_4k()
        self.compass_7k = self.pre_processing_compass_7k()
        self.iris = self.pre_processing_iris()
        self.bc_coimbra = self.pre_processing_bcCoimbra()
        self.credit_approval = self.pre_processing_credit_approval()
        self.user = self.pre_processing_user_knowledge()
        self.stalog_australia = self.pre_processing_stalog_australia()

    # ------------------------------------------------------------------------
    def pre_processing_credit_approval(self):
        df = self.datasets["crx"].copy()
        df = self._prep.label_encoder(df, ["target"])
        df = self._prep.ordinal_encoder(
            df, ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A11", "A12", "A13"]
        )
        df = df.replace("?", np.nan)
        return df.dropna()

    # ------------------------------------------------------------------------
    def pre_processing_user_knowledge(self):
        df = self.datasets["user"].copy()
        df = self._prep.label_encoder(df, ["target"])
        return df

    # ------------------------------------------------------------------------
    def pre_processing_stalog_australia(self):
        df = self.datasets["australian"].copy()
        return df

    # ------------------------------------------------------------------------
    def pre_processing_bcCoimbra(self):
        bc_coimbra = self.datasets["bc_coimbra"].copy()
        bc_coimbra = self._prep.label_encoder(bc_coimbra, ["target"])
        return bc_coimbra

    # ------------------------------------------------------------------------
    def pre_processing_german_credit(self):
        german_credit_df = self.datasets["german"].copy()

        map_gender = {
            "A91": "male",
            "A92": "female",
            "A93": "male",
            "A94": "male",
            "A95": "female",
        }

        german_credit_df["personal-status-and-sex"] = german_credit_df[
            "personal-status-and-sex"
        ].map(map_gender)

        german_credit_df = self._prep.ordinal_encoder(
            german_credit_df,
            [
                "age",
                "checking-account",
                "savings-account",
                "employment-since",
                "telephone",
                "foreign-worker",
                "personal-status-and-sex",
            ],
        )
        german_credit_df = self._prep.label_encoder(german_credit_df, ["target"])

        german_credit_df = self._prep.one_hot_encode(
            german_credit_df,
            [
                "credit-history",
                "purpose",
                "other-debtors",
                "property",
                "other-installment",
                "housing",
                "job",
            ],
        )

        return german_credit_df

    # ------------------------------------------------------------------------
    def pre_processing_adult(self):
        df = self.datasets["adult-clean"].copy()
        df = self._prep.ordinal_encoder(
            df, ["age", "education", "occupation", "gender"]
        )

        df = self._prep.one_hot_encode(
            df,
            ["workclass", "marital-status", "relationship", "native-country", "race"],
        )
        df = self._prep.label_encoder(df, ["target"])
        return df

    # ------------------------------------------------------------------------
    def pre_processing_student_port(self):
        student_port_df = self.datasets["student-por"].copy()
        student_port_df.target = [
            1 if nota >= 10 else 0 for nota in student_port_df.target
        ]
        student_port_df.age = [1 if idade >= 18 else 0 for idade in student_port_df.age]

        student_port_df = self._prep.ordinal_encoder(
            student_port_df,
            [
                "school",
                "sex",
                "address",
                "famsize",
                "Pstatus",
                "schoolsup",
                "famsup",
                "paid",
                "activities",
                "nursery",
                "higher",
                "internet",
                "romantic",
            ],
        )
        student_port_df = self._prep.one_hot_encode(
            student_port_df, ["Mjob", "Fjob", "reason", "guardian"]
        )
        return student_port_df

    # ------------------------------------------------------------------------
    def pre_processing_student_math(self):
        student_mat_df = self.datasets["student-mat"].copy()
        student_mat_df.target = [
            1.0 if nota >= 10 else 0.0 for nota in student_mat_df.target
        ]
        student_mat_df.age = [
            1.0 if idade >= 18 else 0.0 for idade in student_mat_df.age
        ]

        student_mat_df = self._prep.ordinal_encoder(
            student_mat_df,
            [
                "school",
                "sex",
                "address",
                "famsize",
                "Pstatus",
                "schoolsup",
                "famsup",
                "paid",
                "activities",
                "nursery",
                "higher",
                "internet",
                "romantic",
            ],
        )
        student_mat_df = self._prep.one_hot_encode(
            student_mat_df, ["Mjob", "Fjob", "reason", "guardian"]
        )
        return student_mat_df

    # ------------------------------------------------------------------------
    def pre_processing_compass_7k(self):
        compass_7k_df = self.datasets["compas-scores-two-years_clean"].copy()
        clean_compass_7k = compass_7k_df.drop(
            columns=[
                "id",
                "name",
                "first",
                "last",
                "compas_screening_date",
                "dob",
                "days_b_screening_arrest",
                "c_jail_in",
                "c_jail_out",
                "c_case_number",
                "c_offense_date",
                "c_arrest_date",
                "age_cat",
                "vr_case_number",
                "vr_offense_date",
                "decile_score.1",
                "r_case_number",
                "r_offense_date",
                "screening_date",
                "v_screening_date",
                "in_custody",
                "out_custody",
                "priors_count.1",
                "r_jail_in",
                "r_jail_out",
                "vr_charge_degree",
                "vr_charge_desc",
                "v_type_of_assessment",
                "type_of_assessment",
                "violent_recid",
                "r_charge_degree",
                "r_days_from_arrest",
                "c_charge_desc",
                "r_charge_desc",
            ]
        )
        map_races_compass = {
            "African-American": 1,
            "Caucasian": 0,
            "Hispanic": 0,
            "Other": 0,
            "Asian": 0,
            "Native American": 0,
        }
        clean_compass_7k["race"] = clean_compass_7k["race"].map(map_races_compass)

        map_colum = {"two_year_recid": "target"}
        clean_compass_7k = clean_compass_7k.rename(columns=map_colum)
        clean_compass_7k = self._prep.label_encoder(clean_compass_7k, ["target"])
        clean_compass_7k = self._prep.ordinal_encoder(
            clean_compass_7k, ["sex", "c_charge_degree"]
        )
        clean_compass_7k = self._prep.one_hot_encode(
            clean_compass_7k,
            [
                "score_text",
                "v_score_text",
            ],
        )
        return clean_compass_7k

    # ------------------------------------------------------------------------
    def pre_processing_compass_4k(self):
        map_colum = {"two_year_recid": "target"}
        map_races_compass = {
            "African-American": 1,
            "Caucasian": 0,
            "Hispanic": 0,
            "Other": 0,
            "Asian": 0,
            "Native American": 0,
        }
        compass_4k_df = self.datasets["compas-scores-two-years-violent_clean"].copy()
        clean_compass_4k = (
            compass_4k_df.drop(
                columns=[
                    "id",
                    "name",
                    "first",
                    "last",
                    "compas_screening_date",
                    "dob",
                    "days_b_screening_arrest",
                    "c_jail_in",
                    "c_jail_out",
                    "c_case_number",
                    "c_offense_date",
                    "c_arrest_date",
                    "age_cat",
                    "vr_case_number",
                    "vr_offense_date",
                    "decile_score.1",
                    "r_case_number",
                    "r_offense_date",
                    "screening_date",
                    "v_screening_date",
                    "in_custody",
                    "out_custody",
                    "priors_count.1",
                    "r_jail_in",
                    "r_jail_out",
                    "vr_charge_degree",
                    "vr_charge_desc",
                    "v_type_of_assessment",
                    "type_of_assessment",
                    "violent_recid",
                    "r_charge_degree",
                    "r_days_from_arrest",
                    "c_charge_desc",
                    "r_charge_desc",
                ]
            )
            .dropna()
            .reset_index(drop=True)
        )
        clean_compass_4k = clean_compass_4k.rename(columns=map_colum)
        clean_compass_4k["race"] = clean_compass_4k["race"].map(map_races_compass)
        clean_compass_4k = self._prep.label_encoder(clean_compass_4k, ["target"])
        clean_compass_4k = self._prep.ordinal_encoder(
            clean_compass_4k, ["sex", "c_charge_degree"]
        )
        clean_compass_4k = self._prep.one_hot_encode(
            clean_compass_4k, ["score_text", "v_score_text"]
        )
        return clean_compass_4k

    # ------------------------------------------------------------------------
    def pre_processing_wine(self):
        wine_df = self.datasets["wine"].copy()
        wine_df = self._prep.label_encoder(wine_df, ["target"])
        return wine_df

    # ------------------------------------------------------------------------
    def pre_processing_iris(self):
        iris_df = self.datasets["iris"].copy()
        iris_df = self._prep.label_encoder(iris_df, ["target"])
        return iris_df

    # ------------------------------------------------------------------------
    def pre_processing_cervical(self):
        cervical = self.datasets["risk_factors_cervical_cancer"].copy()
        cervical = cervical.drop(
            columns=[
                "STDs: Time since last diagnosis",
                "STDs: Time since first diagnosis",
            ]
        ).replace("?", np.nan)
        return cervical.dropna().astype(float)

    # ------------------------------------------------------------------------
    def pre_processing_chronic(self):
        chronic = self.datasets["kidney_disease"].copy()
        chronic.drop("id", axis=1, inplace=True)
        chronic.columns = [
            "age",
            "blood_pressure",
            "specific_gravity",
            "albumin",
            "sugar",
            "red_blood_cells",
            "pus_cell",
            "pus_cell_clumps",
            "bacteria",
            "blood_glucose_random",
            "blood_urea",
            "serum_creatinine",
            "sodium",
            "potassium",
            "haemoglobin",
            "packed_cell_volume",
            "white_blood_cell_count",
            "red_blood_cell_count",
            "hypertension",
            "diabetes_mellitus",
            "coronary_artery_disease",
            "appetite",
            "peda_edema",
            "aanemia",
            "target",
        ]

        chronic = self._prep.ordinal_encoder(
            chronic,
            [
                "aanemia",
                "peda_edema",
                "appetite",
                "coronary_artery_disease",
                "diabetes_mellitus",
                "hypertension",
                "bacteria",
                "pus_cell",
                "pus_cell_clumps",
                "red_blood_cells",
                "target",
            ],
        )

        return chronic.dropna()

    # ------------------------------------------------------------------------
    def pre_processing_stalog_heart(self):
        stalog = self.datasets["heart"].copy()
        stalog = self._prep.label_encoder(stalog, ["target"])

        return stalog

    # ------------------------------------------------------------------------
    def pre_processing_stroke(self):
        stroke = self.datasets["healthcare-dataset-stroke-data"].copy()
        stroke.drop("id", axis=1, inplace=True)
        stroke = self._prep.ordinal_encoder(
            stroke, ["gender", "ever_married", "smoking_status"]
        )
        stroke = self._prep.one_hot_encode(
            stroke,
            [
                "Residence_type",
                "work_type",
            ],
        )
        return stroke.dropna()

    # ------------------------------------------------------------------------
    def pre_processing_hepatitis(self):
        hepatitis = self.datasets["hepatitis"].copy()
        hepatitis = hepatitis.replace("?", np.nan).dropna()
        hepatitis = self._prep.label_encoder(hepatitis, ["target"])
        return hepatitis.astype("float64")

    # ------------------------------------------------------------------------
    def pre_processing_cleveland(self):
        heart_cleveland = self.datasets["cleveland"].copy()
        heart_cleveland = self._prep.label_encoder(heart_cleveland, ["target"])
        return heart_cleveland

    # ------------------------------------------------------------------------
    def pre_processing_mathernal_rick(self):
        maternal_health_risk_df = self.datasets["Maternal Health Risk Data Set"].copy()
        maternal_health_risk_df = self._prep.label_encoder(
            maternal_health_risk_df, ["target"]
        )
        return maternal_health_risk_df

    # ------------------------------------------------------------------------
    def pre_processing_parkinsons(self):
        parkinsons_df = self.datasets["parkinsons"].copy().drop(columns="name")
        return parkinsons_df

    # ------------------------------------------------------------------------
    def pre_processing_wiscosin(self):
        breast_cancer_wisconsin_df = self.datasets["wiscosin"].copy()
        breast_cancer_wisconsin_df = breast_cancer_wisconsin_df.drop(columns="ID")
        breast_cancer_wisconsin_df = self._prep.label_encoder(
            breast_cancer_wisconsin_df, ["target"]
        )
        return breast_cancer_wisconsin_df

    # ------------------------------------------------------------------------
    def pre_processing_pima(self):
        pima_diabetes_df = self.datasets["pima_diabetes"].copy()
        return pima_diabetes_df

    # ------------------------------------------------------------------------
    def pre_processing_covid(self):
        """
        Method to preprocess the covid dataset

        Returns:
            pd.DataFrame: COVID data
        """
        df = self.datasets["covid"].copy()
        df = df.drop(columns="id_notificacao")
        return df

    # ------------------------------------------------------------------------
    def cria_tabela_sintetico(self):
        tabela_resultados = {}

        syn_cat = pd.read_csv("./data/synthetic/synthetic-cat.csv")
        syn_cont = pd.read_csv("./data/synthetic/synthetic-cont.csv")
        syn_cont_cat = pd.read_csv("./data/synthetic/synthetic-cont-cat.csv")
        syn_one = pd.read_csv("./data/synthetic/synthetic-one.csv")
        syn_two = pd.read_csv("./data/synthetic/synthetic-two.csv")
        syn_three = pd.read_csv("./data/synthetic/synthetic-three.csv")
        syn_rept_one = pd.read_csv("./data/synthetic/synthetic-repeted.csv")
        syn_rept_two = pd.read_csv("./data/synthetic/synthetic-repeted-two.csv")
        syn_rept_three = pd.read_csv("./data/synthetic/synthetic-repeted-three.csv")

        tabela_resultados["datasets"] = [
            syn_cat.astype(float),
            syn_cont.astype(float),
            syn_cont_cat.astype(float),
            syn_one.astype(float),
            syn_two.astype(float),
            syn_three.astype(float),
            syn_rept_one.astype(float),
            syn_rept_two.astype(float),
            syn_rept_three.astype(float)
        ]

        tabela_resultados["nome_datasets"] = [
            "synthetic-cat",
            "synthetic-cont",
            "synthetic-cont-cat",
            "synthetic-one",
            "synthetic-two",
            "synthetic-three",
            "synthetic-repeted-one",
            "synthetic-repeted-two",
            "synthetic-repeted-three" 
        ]
        tabela_resultados["missing_rate"] = [5, 10, 20]

        return tabela_resultados

    # ------------------------------------------------------------------------
    def cria_tabela(self):
        tabela_resultados = {}

        tabela_resultados["datasets"] = [
            # self.pima,
            # self.cleveland,
            # self.wiscosin,
            # self.parkinsons,
            # self.hepatitis,
            # self.mathernal_risk,
            # self.chronic,
            # self.stalog,
            self.cervical,
            self.iris,
            self.wine,
            self.bc_coimbra,
            self.student_math,
            self.student_port,
            self.user,
            self.credit_approval,
            self.german_credit,
            self.compass_4k,
            self.stroke,
            self.compass_7k,
            
        ]

        tabela_resultados["nome_datasets"] = [
            # "pima",
            # "cleveland",
            # "wiscosin",
            # "parkinsons",
            # "hepatitis",
            # "mathernal_risk",
            # "chronic",
            # "stalog",
            "cervical",
            "iris",
            "wine",
            "bc_coimbra",
            "student_math",
            "student_port",
            "user",
            "credit-approval",
            "german-credit",
            "compass-4k",
            "stroke",
            "compass-7k",
            
        ]

        tabela_resultados["missing_rate"] = [5, 10, 20]

        return tabela_resultados
