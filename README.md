# Fraud Detection System

## 1. Project Overview

This project develops a machine learning-based fraud detection system to classify financial transactions as potentially fraudulent or legitimate.

The project covers data exploration, preprocessing, feature engineering, class imbalance handling, model training, evaluation, API development, automated testing, experiment tracking, and cloud deployment.

## 2. Objectives

- Analyze transaction data and identify patterns associated with fraud.
- Train and compare machine learning models.
- Evaluate models using precision, recall, F1-score, ROC-AUC, and Precision-Recall AUC.
- Expose predictions through a FastAPI application.
- Automate code quality checks and tests using GitHub Actions.
- Track experiments and deploy a model using Azure Machine Learning and MLflow.

## 3. Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn and imbalanced-learn
- Matplotlib and Seaborn
- FastAPI and Pydantic
- Pytest and Ruff
- Git and GitHub Actions
- Azure Machine Learning and MLflow

## 4. Repository Structure

- `data/`: Dataset instructions and local data files
- `notebooks/`: Exploratory analysis and experiments
- `src/`: Reusable machine learning code
- `models/`: Local model artifacts
- `api/`: Prediction API
- `mlops/`: Experiment tracking, deployment, and monitoring
- `tests/`: Automated tests
- `screenshots/`: Project evidence
- `.github/workflows/`: CI workflow definitions

## 5. Dataset

The raw dataset is intentionally excluded from Git.

Refer to `data/README.md` for the dataset source, download instructions, and expected file location.

## 6. Local Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies from `requirements.txt`.
4. Follow the instructions in the project documentation and notebooks.

## 7. Evaluation

The models will be evaluated using precision, recall, F1-score, ROC-AUC, Precision-Recall AUC, and confusion matrices. The final model will be selected based on the project's fraud-detection objectives and the trade-off between missed fraud and false alarms.

## 8. Responsible Use

Predictions are intended to support fraud investigation and decision-making. Model limitations, data privacy, potential bias, and the impact of false positives and false negatives must be considered.

## 9. Project Status

Phase 0: Initial project setup.

Further sections will be updated as the project develops.