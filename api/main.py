"""
API de prédiction du prix médian des logements californiens.

Lancer en local (depuis la racine du projet, pas depuis api/) :
    uvicorn api.main:app --reload

Puis ouvrir :
    http://127.0.0.1:8000        -> page de démo interactive (formulaire)
    http://127.0.0.1:8000/docs   -> documentation Swagger générée automatiquement
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Chemins résolus relativement à ce fichier : le serveur fonctionne quel que soit
# le répertoire courant depuis lequel `uvicorn` est lancé.
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="California Housing Price Predictor",
    description=(
        "Prédit la valeur médiane des logements d'un district californien à partir "
        "de variables démographiques et immobilières issues du recensement de 1990. "
        "Modèle : Random Forest optimisé par RandomizedSearchCV (voir le notebook "
        "d'entraînement)."
    ),
    version="1.0.0",
)

# Autorise la page de démo (servie sur le même port) à appeler /predict sans
# blocage CORS. En production réelle, remplacez "*" par le domaine exact du
# front-end plutôt que d'autoriser toutes les origines.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load(MODEL_PATH)
    print(f"Modèle chargé depuis {MODEL_PATH}")
except FileNotFoundError:
    # L'API démarre quand même : /health permet de diagnostiquer le problème
    # sans avoir à lire les logs du serveur.
    model = None
    print(f"ATTENTION : {MODEL_PATH} introuvable. Lancez train_export_model.py d'abord.")


class HousingFeatures(BaseModel):
    """Les 8 variables brutes du recensement, telles qu'attendues par le pipeline.

    Le feature engineering (rooms_per_household, etc.) est recalculé en interne
    par le pipeline chargé -- l'appelant n'a jamais à s'en soucier.
    """

    MedInc: float = Field(
        ..., ge=0,
        description="Revenu médian du district, en dizaines de milliers de dollars",
    )
    HouseAge: float = Field(
        ..., ge=0, le=100,
        description="Âge médian des logements, en années",
    )
    AveRooms: float = Field(
        ..., gt=0,
        description="Nombre moyen de pièces par logement",
    )
    AveBedrms: float = Field(
        ..., gt=0,
        description="Nombre moyen de chambres par logement",
    )
    Population: float = Field(
        ..., ge=0,
        description="Population du district",
    )
    AveOccup: float = Field(
        ..., gt=0,
        description="Nombre moyen d'occupants par logement",
    )
    Latitude: float = Field(
        ..., ge=32.0, le=42.0,
        description="Latitude du district (Californie : environ 32 à 42)",
    )
    Longitude: float = Field(
        ..., ge=-125.0, le=-114.0,
        description="Longitude du district (Californie : environ -125 à -114)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc": 8.3,
                "HouseAge": 25,
                "AveRooms": 6.2,
                "AveBedrms": 1.1,
                "Population": 1200,
                "AveOccup": 3.0,
                "Latitude": 34.1,
                "Longitude": -118.3,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_value_100k: float = Field(
        ..., description="Valeur prédite dans l'unité native du modèle (centaines de milliers de $)"
    )
    predicted_value_usd: float = Field(
        ..., description="Valeur prédite convertie en dollars, pour lisibilité"
    )
    capped_warning: bool = Field(
        ..., description=(
            "True si la prédiction est proche du plafond de censure du dataset "
            "d'entraînement (5.0) -- dans ce cas, le modèle sous-estime probablement "
            "la vraie valeur (voir le notebook, section sur les limites du dataset)."
        )
    )


@app.get("/health", summary="Vérifie que l'API et le modèle sont opérationnels")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, summary="Prédit la valeur médiane des logements")
def predict(features: HousingFeatures) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modèle non chargé. Lancez `python train_export_model.py` à la racine "
                "du projet pour générer model.joblib, puis redémarrez l'API."
            ),
        )

    input_df = pd.DataFrame([features.model_dump()])
    prediction = float(model.predict(input_df)[0])

    return PredictionResponse(
        predicted_value_100k=round(prediction, 4),
        predicted_value_usd=round(prediction * 100_000, 2),
        capped_warning=prediction >= 4.9,
    )


# Sert la page de démo statique à la racine ("/"). Placé en dernier : les routes
# explicites ci-dessus (/health, /predict) sont déjà enregistrées et gardent
# la priorité sur ce montage générique.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="demo")
