from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import torch
import numpy as np
# Assuming your model loading and predictor logic are importable
from utils import AutoRecPredictor, generate_preference_based_recommendations
from preprocessing import device

app = FastAPI(title="AutoRec Inference Service")

# --- Globals to hold in-memory model and data ---
MODEL = None
INTERACTION_MATRIX = None

class PreferenceInput(BaseModel):
    preferences: Dict[str, dict]
    top_k: int = 5

@app.on_event("startup")
async def load_model_and_data():
    """Loads the compiled model and data matrix into memory once at startup."""
    global MODEL, INTERACTION_MATRIX
    # TODO: Load your saved .pth model and interaction matrix here
    # MODEL = torch.load("autorec_model.pth").to(device)
    # INTERACTION_MATRIX = np.load("interaction_matrix.npy")
    print("✅ Model and Matrix loaded into memory.")

@app.post("/predict/existing/{user_id}")
async def predict_existing(user_id: int, top_k: int = 5):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictor = AutoRecPredictor(MODEL, INTERACTION_MATRIX, device)
        recs = predictor.predict_existing_user(user_id, top_k)
        
        # Format the output for the JSON response
        return {"user_id": user_id, "recommendations": [{"item_id": int(i), "score": float(s)} for i, s in recs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/new")
async def predict_new(input_data: PreferenceInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        recs = generate_preference_based_recommendations(
            model=MODEL,
            interaction_matrix=INTERACTION_MATRIX,
            device=device,
            preferences=input_data.preferences,
            top_k=input_data.top_k
        )
        return {"recommendations": [{"item_id": int(i), "score": float(s)} for i, s in recs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))