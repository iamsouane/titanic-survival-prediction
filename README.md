## Titanic Survival Prediction - End-to-End ML Pipeline

## 📌 Project Overview
This project is a complete Machine Learning pipeline to predict passenger survival on the Titanic dataset. 
The main objective was to build a robust, reusable pipeline while strictly avoiding **data leakage**.

## 🛠️ Tech Stack
- **Python 3.12**
- **Pandas & NumPy** (Data manipulation)
- **Scikit-Learn** (Preprocessing, Pipelines, Modeling, Serialization)
- **Matplotlib / Seaborn** (EDA)

## 🧠 Methodology
1. **Exploratory Data Analysis (EDA)**: Handled missing values (`Age`, `Embarked`, `Cabin`) and identified class imbalance (61.6% vs 38.4%).
2. **Data Leakage Prevention**: Used `train_test_split` with `stratify=y` **before** any preprocessing.
3. **Preprocessing Pipeline**: 
   - `ColumnTransformer` to apply specific transformations (Median Imputation + `StandardScaler` for numerical features, Mode Imputation + `OneHotEncoder` for categorical features).
   - Encapsulated inside a Scikit-Learn `Pipeline`.
4. **Modeling**: Compared 3 algorithms using `StratifiedKFold` Cross-Validation (Mean CV Accuracy: ~79.8%):
   - Logistic Regression
   - Decision Tree
   - K-Nearest Neighbors (KNN)
5. **Final Evaluation**: KNN performed best with **82.1% Accuracy** and **84.6% ROC-AUC** on the unseen test set.
6. **Reusable Pipeline**: The preprocessing logic is modularized in `src/`, and the final model is serialized (`.joblib`) for easy reuse.

## 📊 Results
After running the script (`python src/train.py`), here are the results on the unseen test set (20% of the data):

```text
Chargement des données...
Entraînement du modèle...

=== RÉSULTATS ===
Accuracy : 0.8212
ROC-AUC  : 0.8461

Modèle sauvegardé sous : .../knn_titanic_model.joblib

Interpretation:

Accuracy (82.1%): The model correctly predicts survival for 82% of the passengers. It significantly beats the baseline (61.6% by always predicting "not survived").
ROC-AUC (84.6%): The model has a strong ability to rank passengers by survival probability, meaning it can distinguish survivors from non-survivors well.
```

## 📂 Repository Structure
```text
titanic-survival-prediction/
│
├── data/
│   └── raw/                 # Raw CSV files (ignored in git)
│
├── notebooks/
│   └── 01_titanic_pipeline.ipynb  # Jupyter Notebook for EDA and prototyping
│
├── src/
│   ├── data_preprocessing.py      # Preprocessing pipeline (ColumnTransformer)
│   └── train.py                   # Training script (Reusable pipeline)
│
├── .gitignore               # Files to ignore in git
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## 🚀 How to Run

1. Clone the repository
- git clone https://github.com/iamsouane/titanic-survival-prediction.git
- cd titanic-survival-prediction

2. Set up the environment
 - Create a virtual environment
      python -m venv .venv

 - Activate it (On macOS/Linux)
      source .venv/bin/activate  

 - Install dependencies
      pip install -r requirements.txt

3. Run the training pipeline
- python src/train.py

This will train the KNN model, evaluate it, and save the serialized model (knn_titanic_model.joblib) in the root directory.

4. Explore the notebook (Optional)
To see the full Exploratory Data Analysis and model comparison:
- jupyter notebook notebooks/01_titanic_pipeline.ipynb