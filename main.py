from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import io
from typing import List, Dict, Optional
from sklearn.impute import SimpleImputer
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger le modèle et le scaler
try:
    model = joblib.load('modele_faux_billet_26_08_2025.sav')
    scaler = joblib.load('normalisation_26_07_2025.sav')
    logger.info("Modèle et scaler chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle ou du scaler: {e}")
    raise e

# Créer un imputer pour gérer les valeurs manquantes
imputer = SimpleImputer(strategy='mean')

app = FastAPI(
    title="API de Détection de Faux Billets",
    description="API pour la détection de faux billets utilisant un modèle de machine learning",
    version="1.0.0"
)

# Configuration CORS pour permettre les requêtes depuis Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Définir la structure pour une prédiction unique
class BillFeatures(BaseModel):
    diagonal: float
    height_left: float
    height_right: float
    margin_low: float
    margin_up: float
    length: float

# Structure de réponse standardisée
class PredictionResponse(BaseModel):
    prediction: bool
    confidence: float
    probabilities: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    statistics: Dict[str, float]
    chart_data: Dict[str, Dict]
    results: List[Dict]
    sample_size: int

@app.post("/predict", response_model=PredictionResponse)
def predict(bill: BillFeatures):
    """
    Effectue une prédiction sur un seul billet
    """
    try:
        # Convertir les données reçues en DataFrame
        input_data = pd.DataFrame([[
            bill.diagonal,
            bill.height_left,
            bill.height_right,
            bill.margin_low,
            bill.margin_up,
            bill.length
        ]], columns=['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length'])

        # Appliquer la standardisation
        input_data_scaled = scaler.transform(input_data)

        # Faire la prédiction
        prediction = model.predict(input_data_scaled)
        prediction_proba = model.predict_proba(input_data_scaled)

        # Récupérer le résultat
        is_genuine = bool(prediction[0])
        confidence = float(prediction_proba[0][prediction[0]])

        # Formater la réponse
        return PredictionResponse(
            prediction=is_genuine,
            confidence=confidence,
            probabilities={
                "False": float(prediction_proba[0][0]),
                "True": float(prediction_proba[0][1])
            }
        )
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """
    Effectue des prédictions sur un lot de billets via un fichier CSV
    """
    try:
        # Vérifier le type de fichier
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Le fichier doit être au format CSV")
        
        # Lire le fichier CSV
        contents = await file.read()
        
        # Essayer différents séparateurs
        try:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep=';')
        except:
            try:
                df = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep=',')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Erreur de lecture du CSV: {str(e)}")
        
        # Vérifier que les colonnes nécessaires sont présentes
        required_columns = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
        if not all(col in df.columns for col in required_columns):
            return {"error": f"Le fichier doit contenir les colonnes: {required_columns}"}
        
        # Vérifier et gérer les valeurs manquantes
        missing_values = df[required_columns].isnull().sum()
        if missing_values.sum() > 0:
            logger.info(f"Valeurs manquantes détectées: {missing_values.to_dict()}")
            # Remplacer les valeurs manquantes par la moyenne de chaque colonne
            df[required_columns] = df[required_columns].fillna(df[required_columns].mean())
        
        # Vérifier les types de données
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Remplacer les valeurs non numériques par la moyenne
                    df[col] = df[col].fillna(df[col].mean())
                except:
                    raise HTTPException(status_code=400, detail=f"La colonne {col} contient des valeurs non numériques")
        
        # Préparer les données pour la prédiction
        X = df[required_columns]
        
        # Appliquer la standardisation
        X_scaled = scaler.transform(X)
        
        # Faire les prédictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Ajouter les résultats au DataFrame
        df['prediction'] = predictions
        df['confidence'] = probabilities.max(axis=1)
        df['prob_false'] = probabilities[:, 0]
        df['prob_true'] = probabilities[:, 1]
        
        # Calculer les statistiques pour les visualisations
        stats = {
            "total_count": len(df),
            "authentic_count": int(predictions.sum()),
            "fake_count": int(len(df) - predictions.sum()),
            "fake_percentage": float((len(df) - predictions.sum()) / len(df) * 100),
            "avg_confidence": float(probabilities.max(axis=1).mean())
        }
        
        # Préparer les données pour les graphiques
        # Distribution de confiance avec des bins standardisés
        confidence_values = probabilities.max(axis=1)
        bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        counts, _ = np.histogram(confidence_values, bins=bins)
        
        chart_data = {
            "prediction_distribution": {
                "labels": ["Authentiques", "Faux"],
                "values": [stats["authentic_count"], stats["fake_count"]]
            },
            "confidence_distribution": {
                "bins": bins,
                "counts": counts.tolist()
            }
        }
        
        # Convertir le DataFrame en format JSON
        results = df.to_dict(orient='records')
        
        return BatchPredictionResponse(
            statistics=stats,
            chart_data=chart_data,
            results=results,
            sample_size=min(10, len(df))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du traitement par lots: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")

@app.get("/health")
def health_check():
    """
    Endpoint de vérification de la santé de l'API
    """
    return {"status": "healthy", "message": "API de détection de faux billets opérationnelle"}

@app.get("/")
def read_root():
    return {
        "message": "API de détection de faux billets opérationnelle",
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/health",
            "single_prediction": "/predict",
            "batch_prediction": "/predict-batch"
        }
    }

@app.get("/model-info")
def get_model_info():
    """
    Retourne des informations sur le modèle chargé
    """
    return {
        "model_type": str(type(model)),
        "features": ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)