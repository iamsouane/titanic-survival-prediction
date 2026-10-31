import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# On importe nos fonctions depuis le fichier d'à côté
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import get_preprocessor, load_data

def main():
    # 1. Chemin des données (remonte d'un dossier depuis src/ pour aller dans data/raw)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'raw', 'train.csv')
    
    # 2. Chargement
    print("Chargement des données...")
    X, y = load_data(data_path)
    
    # 3. Split (on garde 20% pour le test final)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Création du Pipeline complet (Preprocessing + Modèle)
    model_pipeline = make_pipeline(
        get_preprocessor(),
        KNeighborsClassifier(n_neighbors=5)
    )
    
    # 5. Entraînement
    print("Entraînement du modèle...")
    model_pipeline.fit(X_train, y_train)
    
    # 6. Évaluation
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print("\n=== RÉSULTATS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    
    # 7. SAUVEGARDE (Crucial pour le déploiement)
    model_path = os.path.join(base_dir, '..', 'knn_titanic_model.joblib')
    joblib.dump(model_pipeline, model_path)
    print(f"\nModèle sauvegardé sous : {model_path}")

if __name__ == "__main__":
    main()