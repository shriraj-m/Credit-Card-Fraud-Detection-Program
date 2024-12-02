from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import torch
import xgboost as xgb
from app.models.llm_huggingface import HuggingFaceLLM
from app.models.select_model import ModelSelector

class TransactionRequest(BaseModel):
    features: List[float]
    model_type: str = "xgboost"
    
class PredictionResponse(BaseModel):
    probability: float
    explanation: str
    model_type: str

router = APIRouter()

try:
    # Load XGBoost model
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("app/models/saved_models/xgboost_model.json")

    # Load Pytorch model 
    """
    soon
    """

    # Load Scaler
    scaler = torch.load("app/models/saved_models/scaler.pth")

    # Load feature names
    with open("app/models/saved_models/feature_names.txt", "r") as file:
        feature_names = file.read().split(",")

except Exception as ex:
    print(f"Error loading models: {ex}")
    raise RuntimeError(f"Error loading models: {ex}")


@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    # Model type can be "xgboost" or "pytorch"
    try:
        
        # Prepare input data
        features = np.array(transaction.features).reshape(1, -1)
        scaled_features = scaler.transform(features)
        
        probability = xgb_model.predict_proba(scaled_features)[0][1]
        # Generate summary using Hugging Face LLM
        hf_summarizer = HuggingFaceLLM()
        input_text = f"The probability of fraud is {probability:.4f}. The features are {feature_names}. The model is {transaction.model_type}."
        explanation = hf_summarizer.summarize(input_text)

        return PredictionResponse(
            probability=float(probability), 
            explanation=explanation, 
            model_type=transaction.model_type)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/health")
async def health_check():
    """Check if models are loaded and ready"""
    return {
        "status": "healthy",
        "models_loaded": {
            "xgboost": xgb_model is not None,
            # "pytorch": pytorch_model is not None,
            "scaler": scaler is not None
        }
    }