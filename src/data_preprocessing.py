import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor():
    """
    Crée et retourne le ColumnTransformer pour le dataset Titanic.
    """
    num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
    cat_cols = ['Pclass', 'Sex', 'Embarked']

    # Pipeline numérique : Imputation par la médiane + Normalisation
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline catégoriel : Imputation par le mode + One-Hot Encoding
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Assemblage
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
    
    return preprocessor

def load_data(filepath):
    """
    Charge les données, sélectionne les features et retourne X et y.
    """
    df = pd.read_csv(filepath)
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features]
    y = df['Survived']
    
    return X, y