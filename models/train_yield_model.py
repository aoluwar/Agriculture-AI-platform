import pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from ..features.engineering import add_aggregates

def train_and_save(weather_csv, soil_csv, yield_csv, model_path='models/trained_yield_model.pkl'):
    w=pd.read_csv(weather_csv, parse_dates=['date']); w=add_aggregates(w,10.0,30.0)
    agg=w.agg({'gdd':'sum','precip_mm':'sum','et0_mm':'sum','tmin_c':'mean','tmax_c':'mean'}).to_frame().T
    y=pd.read_csv(yield_csv)
    df=pd.concat([y.reset_index(drop=True), agg.reset_index(drop=True).loc[0:len(y)-1].reset_index(drop=True)], axis=1)
    X=df.drop(columns=['yield_t_ha']); X.fillna(0,inplace=True); yvec=df['yield_t_ha']
    model=RandomForestRegressor(n_estimators=50, random_state=42); model.fit(X, yvec)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, model_path)
    print('saved', model_path)
