import joblib
from pathlib import Path
MODEL_PATH = Path('models/trained_yield_model.pkl')
_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_yield(features: dict):
    m = load_model()
    import pandas as pd
    df = pd.DataFrame([features])
    df = df.fillna(0)
    pred = m.predict(df)[0]
    return float(pred)
