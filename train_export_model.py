"""
Entraîne le pipeline complet (feature engineering + Random Forest) et l'exporte
en un seul fichier joblib, prêt à être chargé par l'API FastAPI.

Usage :
    python train_export_model.py

Produit :
    model.joblib  (à la racine du projet)
"""

import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from model_utils import FeatureEngineer

# Hyperparamètres trouvés dans le notebook via RandomizedSearchCV
# (rf_random_search.best_params_, section 6.2).
#
# ⚠️ REMPLACEZ ces valeurs par celles réellement obtenues chez vous -- elles
# varient selon la graine de la recherche aléatoire et la version de scikit-learn.
# Ce sont des valeurs d'exemple plausibles, pas le résultat réel de votre run.
BEST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
}

RANDOM_STATE = 42
RAW_FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


def main() -> None:
    print("Chargement du dataset California Housing...")
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)[RAW_FEATURE_NAMES]
    y = housing.target

    pipeline = Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("model", RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=-1, **BEST_PARAMS
        )),
    ])

    print("Entraînement du pipeline sur l'intégralité des données...")
    # Contrairement au notebook (qui garde un split train/test pour évaluer le modèle),
    # le modèle de production s'entraîne sur toutes les données disponibles : on ne
    # cherche plus à l'évaluer ici, seulement à exploiter un maximum d'information.
    pipeline.fit(X, y)

    output_path = "model.joblib"
    joblib.dump(pipeline, output_path)
    print(f"Pipeline exporté avec succès -> {output_path}")


if __name__ == "__main__":
    main()
