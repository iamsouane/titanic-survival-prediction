# Titanic Survival Prediction - End-to-End ML Pipeline

## 📌 Project Overview
This project is a complete Machine Learning pipeline to predict passenger survival on the Titanic dataset. 
The main objective was to build a robust, production-ready model while strictly avoiding **data leakage**.

## 🛠️ Tech Stack
- **Python 3.12**
- **Pandas & NumPy** (Data manipulation)
- **Scikit-Learn** (Preprocessing, Pipelines, Modeling)
- **Matplotlib / Seaborn** (EDA)

## 🧠 Methodology
1. **Exploratory Data Analysis (EDA)**: Handled missing values (Age, Embarked, Cabin) and identified class imbalance (61.6% vs 38.4%).
2. **Data Leakage Prevention**: Used `train_test_split` with `stratify=y` **before** any preprocessing.
3. **Preprocessing Pipeline**: 
   - `ColumnTransformer` to apply specific transformations (Median Imputation + StandardScaler for numerical, Mode Imputation + One-Hot Encoding for categorical).
   - Encapsulated inside a Scikit-Learn `Pipeline`.
4. **Modeling**: Compared 3 algorithms using `StratifiedKFold` Cross-Validation:
   - Logistic Regression
   - Decision Tree
   - K-Nearest Neighbors (KNN)
5. **Final Evaluation**: KNN performed best with **81.5% Accuracy** and **82.4% ROC-AUC** on the unseen test set.

## 📂 Repository Structure
```text
├── data/raw/               # Raw CSV files (ignored in git)
├── notebooks/              # Jupyter Notebooks for EDA and prototyping
├── src/                    # Production-ready Python scripts
├── requirements.txt        # Python dependencies
└── README.md

🚀 How to Run

Clone the repository: git clone https://github.com/...
Install dependencies: pip install -r requirements.txt
Run the training script: python src/train.py
