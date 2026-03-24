import pandas as pd
from google import genai
from google.genai import types
from io import StringIO

import re, os
from dotenv import load_dotenv
from openai import OpenAI
from utils.MeLogSingle import MeLogger

import anthropic

_logger = MeLogger()

load_dotenv()

MAPPED_LLMS = {
    "gemini-3-flash-preview": "gemini3",
    "gemini-2.5-flash-lite": "geminiLite",
    "mistralai/devstral-2512": "mistral",
    "xiaomi/mimo-v2-flash": "xiamoi",
    "openai/gpt-4.1-nano":"gpt41nano",
    "gpt-5-mini": "gptMini",
    "gpt-5": "gpt5",
    "anthropic/claude-sonnet-4.5": "claude45",
    "tngtech/tng-r1t-chimera": "deepseek",
    "moonshotai/kimi-k2.5":"kimi"
}

DATASET_NAMES = {
    "pima": "Pima Indians Diabetes",
    "cleveland": "Heart Disease (Cleveland)",
    "wiscosin": "Breast Cancer Wisconsin (Diagnostic)",
    "cervical": "Cervical Cancer (Risk Factors)",
    "parkinsons": "Parkinson's Disease (Voice)",
    "hepatitis": "Hepatitis",
    "chronic": "Chronic Kidney Disease (CKD)",
    "stalog": "Statlog (Heart)",
    "mathernal_risk": "Maternal Health Risk",
    "stroke": "Stroke Prediction Dataset",
    "iris":'Iris',
    "wine":"Wine",
    "bc_coimbra":"Breast Cancer Coimbra",
    "student_math":"Student Performance (Math)",
    "student_port":"Student Performance (Port)",
    "user":"User Knowledge Modeling",
    "credit-approval": "Credit Card Approvals",
    "german-credit":"German Credit",
    "compass-4k":"COMPAS Recidivism",
    "compass-7k":"COMPAS viol Recidivism",
    # Dataset Sintético
    "synthetic-cont-cat": "Synthetic",
    "synthetic-cat": "Synthetic",
    "synthetic-cont": "Synthetic",
    "synthetic-one": "Synthetic",
    "synthetic-two": "Synthetic",
    "synthetic-three": "Synthetic",
    "synthetic-repeted-two": "Synthetic",
    "synthetic-repeted-three": "Synthetic",
    "synthetic-repeted" : "Synthetic"

}

def tratar_dados_infor(data_array):
    # 1. Extrair as strings de dentro das sub-listas e criar uma lista simples
    # data_array[:, 0] pega o conteúdo de cada sub-array
    raw_strings = data_array.flatten()
    
    # 2. Dividir as strings pela vírgula
    # Isso cria uma lista de listas: [['120', '0.00', ...], ['121', ...]]
    split_data = [line.split(',') for line in raw_strings]
    
    # 3. Criar o DataFrame
    df = pd.DataFrame(split_data)
    df = df.drop(columns=df.columns[0])
    
    # 4. Converter tudo para numérico (as colunas vêm como strings)
    # pd.to_numeric com errors='coerce' ajuda se houver algum lixo nos dados
    df = df.apply(pd.to_numeric, errors='ignore')
    
    return df

def clean_and_parse_llm_data(response_text, expected_shape):
    # 1. Extração via Regex (Limpa as "conversas" da LLM)
    match = re.search(r"```(?:csv)?\s*(.*?)\s*```", response_text, re.DOTALL)
    content = match.group(1).strip() if match else response_text.strip()

    # 2. Estratégia de tentativa e erro (CSV vs Espaços)
    # Tentamos primeiro o separador que você definiu no novo prompt (Vírgula)
    for separator in [",", r"\s+"]:
        try:
            df_imputed = pd.read_csv(StringIO(content), sep=separator, engine="python")
            return df_imputed
        except Exception:
            continue

    print(response_text)
    raise ValueError(f"Não foi possível parsear os dados. Esperado {expected_shape}.")


def adjust_prompt(dataset_name: str, missing_data: pd.DataFrame):
    headers_str = ", ".join(missing_data.columns)
    string_missing = missing_data.to_string()
    prompt = f"""
    You are an expert data analyst. I am providing a subset of the {dataset_name} Dataset.
    Task: Use your knowledge of this specific dataset's statistical properties (feature ranges, class distributions, and correlations) to perform data imputation.
    Constraint: DO NOT execute Python code. DO NOT provide any conversational text. Do NOT return any NaN or ? value.

    The matrix below contains missing values. Impute them to be as consistent as possible with the original dataset.
    Matrix:
    {string_missing}

    Output Format:
    Return the complete imputed matrix inside a single Markdown code block. 
    Use CSV format (comma-separated values) with the original headers.
    Expected Columns ({missing_data.shape[1]}): 
    [{headers_str}]

    Strict Rules:
    1. Start directly with the code block: ```csv
    2. End exactly with: ```
    3. Ensure the exact same number of rows as the input.
    4. No explanations, no introductory text, no "Here is the matrix".
    5. Use commas as delimiters. Every row MUST have exactly {missing_data.shape[1] - 1} commas.
    """
    return prompt


def llm_impute(
    dataset_name: str,
    X_teste_norm_md: pd.DataFrame,
    model_name: str,
    api: str = "open_router",
) -> str:
    """
    Método para realizar a imputação com Large Language Models (LLMs),
    utilizando a API do Gemini ou OpenRouter

    Args:
        - dataset_name (str)
        - X_teste_norm_md (pd.DataFrame)
        - model_name (str)
        - api (str)
    """
    try:
        match api:
            case "open_router":
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("API_KEY_OPEN_ROUTER"),
                )

            case "gemini":
                client = genai.Client(
                    api_key=os.getenv("API_KEY_GEMINI"),
                    http_options={"timeout": 10 * 60 * 1000},
                )
            case "gpt":
                client = OpenAI(
                    api_key=os.getenv("API_KEY_GPT"),
                )
            case "claude":
                client = anthropic.Anthropic(api_key=os.getenv("API_KEY_CLAUDE"))

        output = X_teste_norm_md.copy()

        batch_row = 40
        batch_col = 10
        iter_batch = 0
        row_start = col_start = 0  # initialised before loop for safe error reporting

        n_rows, n_cols = X_teste_norm_md.shape

        for row_start in range(0, n_rows, batch_row):
            row_end = min(row_start + batch_row, n_rows)
            actual_start = row_start
            if (row_end - row_start) < batch_row and n_rows >= batch_row:
                actual_start = row_end - batch_row

            for col_start in range(0, n_cols, batch_col):
                col_end = min(col_start + batch_col, n_cols)
                batch_to_prompt = X_teste_norm_md.iloc[
                    actual_start:row_end, col_start:col_end
                ]
                _logger.info(f"Batch = {iter_batch}")
                match api:
                    case "open_router":

                        response = client.responses.create(
                            model=model_name,
                            temperature=0.05,
                            input=adjust_prompt(
                                dataset_name=dataset_name, missing_data=batch_to_prompt
                            ),
                        )
                        imputed_value_str = response.output[0].content[0].text

                    case "gpt":

                        response = client.responses.create(
                            model=model_name,
                            # tools=[{"type": "web_search"}],
                            input=adjust_prompt(
                                dataset_name=dataset_name, missing_data=batch_to_prompt
                            ),
                        )
                        imputed_value_str = response.output_text

                    case "gemini":
                        grounding_tool = types.Tool(google_search=types.GoogleSearch())
                        response = client.models.generate_content(
                            model=model_name,
                            contents=adjust_prompt(
                                dataset_name=dataset_name, missing_data=batch_to_prompt
                            ),
                            config=types.GenerateContentConfig(
                                temperature=0.1,
                                thinking_config=types.ThinkingConfig(thinking_budget=0),
                                # tools=[grounding_tool])
                            ),
                        )

                        imputed_value_str = response.text.strip()

                    case "claude":
                        response = client.messages.create(
                            model=model_name,
                            max_tokens=10000,
                            messages=[
                                {
                                    "role": "user",
                                    "content": adjust_prompt(
                                        dataset_name=dataset_name,
                                        missing_data=batch_to_prompt,
                                    ),
                                }
                            ],
                            temperature=0.1,
                        )
                        imputed_value_str = response.content[0].text

                # Converte CSV retornado pela LLM em DataFrame
                df_imputed = clean_and_parse_llm_data(
                    response_text=imputed_value_str,
                    expected_shape=batch_to_prompt.shape,
                )

                rows_needed = row_end - row_start
                clean_imputed_data = df_imputed.iloc[-rows_needed:, :]
                # Escreve no output
                actual_rows = clean_imputed_data.shape[0]

                for col in clean_imputed_data.columns:
                    if col not in output.columns:
                        clean_imputed_data = clean_imputed_data.drop(columns=col)

                if actual_rows != rows_needed:
                    # Caso retorne um shape diferente do esperado, retorna os valores originais
                    # serão preenchidos com zero
                    _logger.warning("LLM provide an differ dataframe shape")
                    output.iloc[row_start:row_end, col_start:col_end] = output.iloc[
                        row_start:row_end, col_start:col_end
                    ].values

                elif clean_imputed_data.empty:
                    _logger.warning("LLM provide an empty dataframe")
                    output.iloc[row_start:row_end, col_start:col_end] = output.iloc[
                        row_start:row_end, col_start:col_end
                    ].values

                else:
                    try:
                        _logger.warning("LLM perform imputation correctly")
                        output.iloc[row_start:row_end, col_start:col_end] = (
                            clean_imputed_data.values
                        )
                    except Exception:
                        _logger.warning("LLM provide an dataframe that must be treated")
                        df_tratado = tratar_dados_infor(clean_imputed_data.values)
                        output.iloc[row_start:row_end, col_start:col_end] = (
                            df_tratado.values
                        )

                iter_batch += 1

    except Exception as e:
        _logger.error(
            f"Erro no batch [{row_start}:{row_end}, {col_start}:{col_end}]: {e}"
        )
        raise ValueError(e)

    # Lógica Similar a Pré-Imputação

    if output.isna().any().any():
        _logger.warning("LLM failed to impute some values. Applying fallback...")
        # Caso tenha NaN, substituir pela média

        for col in output.columns:
            if output[col].isna().any():

                mean_val = output[col].astype(float).mean()
                # If the whole column is NaN (LLM failed entirely), use 0
                fill_val = mean_val if not pd.isna(mean_val) else 0.5
                
                output[col] = output[col].fillna(fill_val)

        # Para um framework, podemos adotar um Imputador (ex: MICE)
        # Para performar imputação aqui

    return output

class LLMWrapper:
    """A minimal wrapper to make llm_impute compatible with sklearn-style fit/transform."""
    def __init__(self, model_name: str, api: str, dataset_name: str, feature_names: list[str]):
        self.model_name = model_name
        self.api = api
        self.dataset_name = dataset_name
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        df_imputed = llm_impute(
            dataset_name=self.dataset_name,
            X_teste_norm_md=X,
            model_name=self.model_name,
            api=self.api
        )
        return df_imputed.values
