# Retrieval-Augmented Generation for Missing Data Imputation in Tabular Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the codebase for the paper: *TabRAG-XAI-Imputer: An Explainable Retrieval-Augmented Framework for LLM-Based Tabular Missing Data Imputation*

## Paper Details
- Authors: Arthur Dantas Mangussi, Ricardo Cardoso Pereira, Miriam Seone Satnos, Ana Carolina Lorena, Mykola Pechenizkiy, and Pedro Henriques Abreu
- Abtract:Missing data is a pervasive challenge in real-world tabular datasets,often leading to biased analyses and degraded machine learning performance. Existing imputation methods rely on distributional assumptions, local
distance heuristics, or generative adversarial training, none of which explicitly leverage the local relational structure of the data during inference. In this work, we propose \textbf{TabRAG-XAI-Imputer}, a novel framework that integrates a correlation-weighted retrieval mechanism with the generative capabilities of Large Language Models (LLMs) through a Retrieval-Augmented Generation (RAG) pipeline. By retrieving the most relevant complete records at inference time, the proposed framework grounds LLM predictions in local data context, improving the reliability and accuracy of imputed values, particularly in scenarios where dataset-specific patterns are unlikely to be represented in the model's pretraining corpus. In addition to performing data imputation, \textbf{TabRAG-XAI-Imputer} provides instance-level explanations for imputed values from an Explainable Artificial Intelligence (XAI) perspective, enhancing transparency and interpretability in the imputation process. We evaluate the proposed framework on 20 tabular datasets under different missing data mechanisms and missing rates of 5\%, 10\%, and 20\%. Experimental results show that \textbf{TabRAG-XAI-Imputer} achieves the lowest overall Mean Absolute Error (MAE) under Missing Not At Random (MNAR) settings and ranks second under Missing At Random (MAR) settings. Furthermore, evaluations on downstream machine learning tasks demonstrate that the proposed method preserves data utility and yields competitive predictive performance. These findings highlight the potential of retrieval-augmented LLMs as an effective and interpretable approach for missing data imputation.
- Keywords: Missing Data Imputation; Large Language Models; Retrieval-Augmented Generation; Tabular Data
- Year: 2026
- Contact: mangussiarthur@gmail.com

## Installation
```bash
git clone https://github.com/ArthurMangussi/RAGImputation.git
cd RAGImputation
pip install -r requirements.txt
```
## Citation
```bash
As soon as possible
```