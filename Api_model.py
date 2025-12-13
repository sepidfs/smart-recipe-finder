# Api_model.py
from pathlib import Path
import json
import joblib

# ---------- locate model safely ----------
HERE = Path(__file__).resolve().parent
CANDIDATES = [
    HERE / "cuisine_pipeline.joblib",            # same folder as API
    HERE.parent / "cuisine_pipeline.joblib",     # parent folder (your screenshot)
    Path.cwd() / "cuisine_pipeline.joblib",      # current working dir
]
MODEL_PATH = next((p for p in CANDIDATES if p.exists()), None)
if MODEL_PATH is None:
    raise FileNotFoundError(
        f"cuisine_pipeline.joblib not found. Looked in: "
        + ", ".join(str(p) for p in CANDIDATES)
    )

pipeline = joblib.load(MODEL_PATH)

# Optional: labels file (اگر داری)
LABELS_PATHS = [
    HERE / "labels.json",
    HERE.parent / "labels.json",
]
label_names = None
for lp in LABELS_PATHS:
    if lp.exists():
        try:
            label_names = json.loads(lp.read_text(encoding="utf-8"))
        except Exception:
            label_names = None
        break

def predict_cuisine(ingredients, diet="none", top_k=3, notes=""):
    """
    ingredients: list[str]
    diet: "none" | "vegetarian" | "vegan" | ...
    """
    if not isinstance(ingredients, list) or not ingredients:
        return {"error": "ingredients must be a non-empty list of strings"}

    # Join into a single text sample for TF-IDF
    text = " ".join(str(x) for x in ingredients)
    probs = pipeline.predict_proba([text])[0]   # shape (K,)
    classes = pipeline.classes_                 # numpy array of class indices or names

    # top-k
    import numpy as np
    k = max(1, min(top_k, len(probs)))
    idx = np.argsort(probs)[::-1][:k]
    top = []
    for i in idx:
        cname = classes[i]
        # map with label_names if provided (optional)
        if label_names and str(cname) in label_names:
            cname = label_names[str(cname)]
        top.append({"label": str(cname), "prob": float(probs[i])})

    return {
        "top_k": k,
        "predictions": top,
        "diet": diet,
        "notes": notes,
        "model_path": str(MODEL_PATH),
    }
