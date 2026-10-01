# California Housing — API de prédiction

Déploiement du modèle Random Forest entraîné dans [`california_housing_optimized.ipynb`](../california_housing_optimized.ipynb),
sous forme d'une API FastAPI avec page de démo interactive.

## Structure

```
california_housing_api/
├── model_utils.py          # Transformation de feature engineering partagée
├── train_export_model.py   # Entraîne et exporte le pipeline complet (model.joblib)
├── requirements.txt
├── api/
│   ├── main.py              # Application FastAPI
│   └── static/
│       └── index.html       # Page de démo (formulaire + appel à /predict)
└── model.joblib             # Généré par train_export_model.py (pas versionné sur Git)
```

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Mettre à jour les hyperparamètres

Ouvrez `train_export_model.py` et remplacez `BEST_PARAMS` par les valeurs réellement
obtenues dans votre notebook (`rf_random_search.best_params_`, section 6.2). Les valeurs
fournies par défaut sont des exemples plausibles, pas le résultat de votre entraînement.

## 2. Entraîner et exporter le modèle

```bash
python train_export_model.py
```

Ceci génère `model.joblib` à la racine du projet (quelques dizaines de secondes).

## 3. Lancer l'API

Depuis la racine du projet (important : pas depuis `api/`, sinon les imports relatifs échouent) :

```bash
uvicorn api.main:app --reload
```

Puis ouvrez :
- **http://127.0.0.1:8000** — page de démo interactive avec formulaire
- **http://127.0.0.1:8000/docs** — documentation Swagger générée automatiquement par FastAPI

## Exemple d'appel direct (curl)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3,
    "HouseAge": 25,
    "AveRooms": 6.2,
    "AveBedrms": 1.1,
    "Population": 1200,
    "AveOccup": 3.0,
    "Latitude": 34.1,
    "Longitude": -118.3
  }'
```

## Pourquoi un `Pipeline` plutôt qu'un modèle seul ?

Le dataset brut ne contient que 8 variables (`MedInc`, `HouseAge`, etc.) ; le notebook
calcule 3 variables dérivées supplémentaires (`rooms_per_household`, `bedrooms_per_room`,
`population_per_household`) avant l'entraînement. Si l'API recevait un modèle entraîné
seul, il faudrait **dupliquer ce calcul en Python dans l'API** — source classique de bugs
silencieux si le notebook évolue sans que l'API suive.

En embarquant le feature engineering (`model_utils.FeatureEngineer`) et le modèle dans un
seul `Pipeline` scikit-learn sérialisé par `joblib`, l'API reste une simple coquille :
elle charge un objet, lui passe les données brutes, et appelle `.predict()`. Toute la
logique métier vit à un seul endroit.

## Limites connues (à mentionner si vous présentez cette démo)

- Modèle entraîné sur des données de recensement de **1990** : les montants ne reflètent
  pas les prix immobiliers actuels.
- `MedHouseVal` est plafonné à 5.00001 (500 001 $) dans les données d'entraînement : les
  prédictions proches de ce seuil sous-estiment probablement la vraie valeur (l'API
  renvoie un champ `capped_warning` pour signaler ce cas).
- Démo à but pédagogique, pas un outil d'estimation immobilière réel.
