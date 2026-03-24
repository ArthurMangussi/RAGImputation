# RAGImputation

A robust framework for Missing Data Imputation utilizing both Traditional Machine Learning approaches, LLM-based approaches, and a novel Retrieval-Augmented Generation (RAG) methodology.

This project is part of a PhD research investigating the effectiveness of semantic embeddings, vector databases (FAISS), and Large Language Models (LLMs) in replacing traditional distance-based algorithms like KNN and MICE.

## 🚀 Features

- **Standardized Imputation Benchmarking:**
  - Evaluates models across various missing data mechanisms (MCAR, MAR, MNAR) and missing rates.
  - Automatically handles K-Fold cross-validation, feature normalization, and metric generation (MAE / NRMSE).
  
- **Multiple Imputers Supported via `ModelsImputation`:**
  - `mice`, `knn`, `missforest`, `softimpute`, `gain`, `tabpfn`, `saei`, `pmivae`
  - Multiple **pure LLMs** via OpenRouter/Gemini (Claude, GPT, Gemini, Llama, Mixtral, etc.).
  - **`rag`**: A novel `RAGImputer` using `sentence-transformers` to map dataset rows into natural language, storing complete rows in a `faiss` index for semantic similarity search, and aggregating or prompt-injecting exact matches.

## 🛠 Installation

This project relies on several specialized libraries (PyTorch, TensorFlow, FAISS, Sentence-Transformers, HuggingFace Hub, etc). 

To ensure exact reproducibility, it is highly recommended to use the provided Conda environment script (`environment.yml`).

### Option 1: Using Conda (Recommended)

1. Clone the repository and navigate into the directory.
2. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```bash
   conda activate RAGImputation
   ```

### Option 2: Using Pip

If you do not use Conda, you can use raw `pip` (Python 3.11.9 is recommended):

```bash
python -m venv .venv
# Activate the venv (Windows: .venv\Scripts\activate | Unix: source .venv/bin/activate)
pip install -r requirements.txt
```

## ⚙️ Configuration (.env)

If you are using LLMs or the RAG (`mode="llm"`), be sure to configure your API keys in the `.env` root file:

```env
API_KEY_OPEN_ROUTER=your_open_router_key_here
API_KEY_GEMINI=your_gemini_key_here
API_KEY_GPT=your_openai_key_here
API_KEY_CLAUDE=your_anthropic_key_here
```

## 🧪 Usage

You can trigger distinct experimental designs using the scripts inside the `codes` directory:

1. Validate pure **MCAR** mechanism:
```bash
python codes/experimental_design_llm_mcar.py
```

2. Validate pure **MAR** mechanism:
```bash
python codes/experimental_design_llm_mar.py
```

3. Validate pure **MNAR** mechanism:
```bash
python codes/experimental_design_llm_mnar.py
```

All results will be automatically logged to the terminal and stored as tabular CSV files inside the `results/` folder, organized securely by model name, missing rate, mechanism, and fold index.

## 🏗 Project Architecture

```plaintext
RAGImputation/
├── algorithms/
│   ├── llm.py           # LLMWrapper & OpenRouter/Gemini API integrators
│   ├── rag_imputer.py   # Sentence-Transformers + FAISS RAG model
│   ├── gain.py          # GAN architecture models
│   ├── pmivae.py        # Partial VAE variants
│   └── saei.py          # Siamese Autoencoders
├── codes/               # The experimental design (MCAR, MAR, MNAR) benchmarking runners
├── data/                # Datasets storage
├── results/             # Auto-generated prediction results and missing data grids
├── utils/
│   ├── MyModels.py          # Central factory for loading *all* models reliably
│   ├── MyPreprocessing.py   # Auto-scalers & normalizers
│   ├── MyResults.py         # Automated MAE/NRMSE evaluators without data leakage
│   └── MeLogSingle.py       # Custom unified logger
├── .env                 # API Storage Keys
├── environment.yml      # Conda Setup Script
└── requirements.txt     # Complete raw dependency snapshot
```

## ✉️ Author
**Arthur Dantas Mangussi**  
Aeronautics Institute of Technologies (ITA) - Brazil  
University of Coimbra (UC) - Portugal  
`mangussiarthur@gmail.com`