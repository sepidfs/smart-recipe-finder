# app.py  (run with: uvicorn app:app --reload)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import traceback

from Api_model import predict_cuisine  # imports pipeline once

app = FastAPI(title="Smart Recipe API", version="0.1.0")

# --- CORS (allow local HTML/JS to call the API) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # for demo; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictIn(BaseModel):
    ingredients: List[str] = Field(..., description="List of ingredient tokens")
    diet: str = Field("none")
    top_k: int = Field(3, ge=1, le=5)
    notes: str = Field("", description="Optional notes")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictIn):
    try:
        out = predict_cuisine(req.ingredients, req.diet, req.top_k, req.notes)
        if isinstance(out, dict) and "error" in out:
            raise HTTPException(status_code=422, detail=out["error"])
        return out
    except HTTPException:
        raise
    except Exception as e:
        # Return useful detail during local dev
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{e.__class__.__name__}: {e}\n{tb}")
