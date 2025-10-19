# Onco-Logic: An Integrated AI Suite for Precision Oncology

<!-- Core -->
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)

<!-- ML / NLP -->
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white)

<!-- Data / Viz -->
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4AABCF?logo=seaborn&logoColor=white)

<!-- Classical/Boosted Models (no official logos in Simple Icons) -->
![XGBoost](https://img.shields.io/badge/XGBoost-8A2BE2)
![LightGBM](https://img.shields.io/badge/LightGBM-67A926)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00)
![imbalanced--learn](https://img.shields.io/badge/imbalanced--learn-005C97)
![UMAP](https://img.shields.io/badge/UMAP-008080)
![SHAP](https://img.shields.io/badge/SHAP-CC0000)

<!-- Infra -->
![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)
![Git%20LFS](https://img.shields.io/badge/Git%20LFS-F05032?logo=git&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-000000)

---

## Overview

Onco-Logic is a comprehensive, multi-modal decision support ecosystem designed to transform cancer care by unifying fragmented patient data. The suite leverages advanced AI and machine learning to provide clinicians and researchers with a holistic understanding of each patient's disease, enabling a new frontier in precision oncology.

The project is built on three synergistic pillars, each targeting a critical data modality in cancer research and clinical practice:

  * **Pathfinder-NLP**: An NLP pipeline that extracts and standardizes critical information from unstructured, free-text pathology reports using a fine-tuned ClinicalBERT model. It transforms narrative documents into structured, computable data for large-scale analysis and clinical decision support.
  * **Geno-Classify**: A machine learning framework that analyzes complex gene expression profiles to perform precise, automated cancer subtyping. It addresses the "high-dimension, low-sample-size" challenge to guide targeted therapies and uncover biological insights.
  * **Pred-Nosis**: A predictive modeling tool that forecasts breast cancer patient survival status (Alive/Deceased) and duration. It employs advanced data resampling techniques (SMOTE+ENN) to handle severe class imbalance and achieve high-accuracy predictions.

This repository contains the complete Streamlit application, which provides an interactive interface for exploring datasets, evaluating models, and utilizing the predictive capabilities of each pillar.

-----

## Tech Stack

  * **Backend & ML:** Python 3.9+, PyTorch, Scikit-learn, Transformers (Hugging Face), SHAP, UMAP, Imbalanced-learn
  * **Frontend:** Streamlit
  * **Data Manipulation:** Pandas, NumPy
  * **Plotting & Visualization:** Matplotlib, Seaborn
  * **NLP:** SpaCy, NLTK
  * **LLM Integration:** OpenRouter API for generating prognostic summaries
  * **Environment Management:** `dotenv` for managing API keys

-----

## Features & Application Pages

The Onco-Logic suite is delivered as an interactive Streamlit application with the following key modules:

### 1\. NLPathology (Pathfinder-NLP)

An end-to-end NLP pipeline for processing and analyzing oncological pathology reports.

  * **Interactive Report Processor**: Upload or paste raw pathology text to automatically extract structured data, including cancer type and TNM staging, using a fine-tuned ClinicalBERT model.
  * **AI-Powered Summaries**: Integrates with the OpenRouter API to generate professional, medically accurate prognostic summaries based on the extracted clinical information.
  * **Unsupervised Text Exploration**: Utilizes TF-IDF vectorization with PCA, t-SNE, and UMAP to visualize high-dimensional text data and reveal inherent clustering of reports by cancer type.
  * **N-gram Frequency Analysis**: Identifies the most common bi-grams and tri-grams in the text corpus to uncover key clinical terminology and patterns.
  * **Named Entity Recognition (NER)**: A demonstration using SpaCy to automatically highlight and classify clinical entities within pathology reports.

### 2\. CancerGenetics (Geno-Classify)

A comprehensive toolkit for cancer subtype classification based on gene expression data.

  * **Clustering Analysis**: Employs K-Means clustering on data processed through various feature engineering pipelines (e.g., Variance Thresholding, UMAP, Log Transformation) to evaluate unsupervised subtype separation.
  * **Model Evaluation**: Trains and evaluates multiple classifiers (Decision Tree, Random Forest, SGD Classifier) using stratified cross-validation to predict cancer subtypes with high accuracy.
  * **SHAP Feature Importance**: Uses SHAP (SHapley Additive exPlanations) to interpret model predictions and identify the most influential genes driving classification, bridging the gap between prediction and biological insight.
  * **Clinical Predictor Tool**: An interactive tool to predict cancer subtypes based on user-inputted gene expression values, leveraging the best-performing trained models.

### 3\. BreastCancerSurvival (Pred-Nosis)

A robust application for predicting breast cancer patient outcomes using structured clinical data.

  * **Multi-Class Survival Prediction**: Forecasts survival duration across multiple time-window categories (e.g., \<1 year, 1-2 years) with high accuracy.
  * **Advanced Resampling**: Implements a state-of-the-art hybrid resampling technique (SMOTE+ENN) to address severe class imbalance in survival data, boosting Random Forest model accuracy to 95.9%.
  * **Model Bake-off**: Systematically trains, evaluates, and compares multiple models (Random Forest, XGBoost, CatBoost, LightGBM) on raw, oversampled, and cleaned datasets.
  * **Interactive Risk Predictor**: Allows users to input patient characteristics and receive a survival risk prediction, complete with model confidence scores.

-----

## Project Structure

```text
Onco-Logic/
├── App.py                    # Main Streamlit application launcher
├── requirements.txt          # Project dependencies
├── .env                      # Environment variables for API keys
├── .env.example              # Example environment file
├── .gitignore                # Specifies files and directories for Git to ignore
├── .gitattributes            # Configures Git Large File Storage (LFS) for large files
├── Data/                     # Directory for datasets
├── models/                   # Directory for saved models
├── pages/
│   ├── BreastCancerSurvival.py # Streamlit page for the "Pred-Nosis" module
│   ├── CancerGenetics.py       # Streamlit page for the "Geno-Classify" module
│   ├── NLPathology.py          # Streamlit page for the "Pathfinder-NLP" module
│   └── Secondary Endpoint/
│       └── BreastCancerStatus.py # Additional analysis page
├── resources/                # Directory for additional resources (for cancer genetics models)
└── scripts/
    ├── cancer_trainer.py       # Script for training the cancer type classification model
    ├── stage_trainer.py        # Script for training the TNM staging models
    ├── llm_api.py              # Module for interacting with the OpenRouter LLM API
    ├── custom_transformers.py  # Custom scikit-learn transformers for ML pipelines
    └── __pycache__/            # Cache directory for compiled Python files
        ├── custom_transformers.cpython-313.pyc
        └── llm_api.cpython-313.pyc
```

-----

## Quick Start (Local)

### 1\. Prerequisites

  * Python 3.9+ and `pip`
  * Git and Git LFS for handling large model files

### 2\. Clone & Install

```bash
git clone <this-repo-url>
cd Onco-Logic
pip install -r requirements.txt
```

### 3\. Configure Environment

Copy the example environment file and add your OpenRouter API key to enable the AI-powered prognosis summary feature in the NLPathology tool.

```bash
cp .env.example .env
# Then edit .env and add your key
```

Required keys:

  * `OPENROUTER_API_KEY`: Your API key from [OpenRouter.ai](https://openrouter.ai/).

### 4.\. Datasets and Models

This repository includes the datasets and pre-trained models required to run the application.

* **Datasets**: The following CSV files are located in the `Data/` directory:
    * `cancer_subtype_data.csv`
    * `cancer_subtype_labels.csv`
    * `Breast_Cancer.csv`
    * `TCGA_M01_patients.csv`
    * `TCGA_N03_patients.csv`
    * `TCGA_patient_to_cancer_type.csv`
    * `TCGA_Reports.csv`
    * `TCGA_T14_patients.csv`
* **Models**: The pre-trained NLP models are included in the `models/` directory. You can also retrain them using the provided scripts:
    * Run `scripts/cancer_trainer.py` to train the cancer type classification model.
    * Run `scripts/stage_trainer.py --stage all` to train the T, N, and M staging models.

### 5\. Run the App

Once the environment is configured and datasets are in place, run the Streamlit application:

```bash
streamlit run App.py
```

The application will be available at `http://localhost:8501`. Use the sidebar to navigate between the different modules of the Onco-Logic suite.