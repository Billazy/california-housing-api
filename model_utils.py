"""
Transformation de feature engineering partagée entre l'entraînement et l'API.

IMPORTANT : cette classe doit être importée depuis EXACTEMENT ce même module
(`model_utils.py`) au moment de l'entraînement (train_export_model.py) et au
moment du chargement par l'API (api/main.py). joblib sérialise une référence
au chemin du module + nom de classe, pas le code lui-même -- si la classe est
redéfinie ailleurs (ex. copiée-collée dans le notebook), joblib.load() échouera
côté API avec `AttributeError: Can't get attribute 'FeatureEngineer'`.
"""

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Calcule les 3 variables dérivées utilisées à l'entraînement du modèle.

    Reçoit les 8 variables brutes du dataset California Housing et ajoute :
    - rooms_per_household        = AveRooms / AveOccup
    - bedrooms_per_room          = AveBedrms / AveRooms
    - population_per_household  = Population / AveOccup
    """

    def fit(self, X, y=None):
        # Rien à apprendre : ce sont des calculs déterministes, pas des statistiques
        # ajustées sur les données (contrairement à StandardScaler, par exemple).
        return self

    def transform(self, X):
        X = X.copy()
        X["rooms_per_household"] = X["AveRooms"] / X["AveOccup"]
        X["bedrooms_per_room"] = X["AveBedrms"] / X["AveRooms"]
        X["population_per_household"] = X["Population"] / X["AveOccup"]
        return X
