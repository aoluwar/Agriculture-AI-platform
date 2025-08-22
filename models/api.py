from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from .data_ingest.weather_api import fetch_weather_nasa
from .features.engineering import add_aggregates
from .models.predict import predict_yield
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = FastAPI(title='AgriAI Platform', version='3.0')

class YieldRequest(BaseModel):
    gdd_sum: float; precip_mm_sum: float; et0_mm_sum: float; tmin_c_mean: float; tmax_c_mean: float

class WeatherForecastRequest(BaseModel):
    lat: float; lon: float; start: str; end: str; variables: list; horizon: int = 14

@app.get('/health')
def health(): return {'status':'ok'}

@app.post('/yield/predict')
def yield_predict(req: YieldRequest):
    features = {'gdd': req.gdd_sum, 'precip_mm': req.precip_mm_sum, 'et0_mm': req.et0_mm_sum, 'tmin_c': req.tmin_c_mean, 'tmax_c': req.tmax_c_mean}
    return {'predicted_yield_t_ha': predict_yield(features)}

@app.post('/weather/forecast')
def weather_forecast(req: WeatherForecastRequest):
    df = fetch_weather_nasa(req.lat, req.lon, req.start, req.end)
    df = df.sort_values('date')
    out = {}
    for var in req.variables:
        if var not in df.columns:
            out[var] = []
            continue
        series = df.set_index('date')[var].astype(float).dropna()
        if len(series) < 10:
            out[var] = [float(series.iloc[-1]) for _ in range(req.horizon)]
            continue
        try:
            model = SARIMAX(series, order=(1,0,0), seasonal_order=(0,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            pred = res.get_forecast(steps=req.horizon).predicted_mean
            out[var] = [float(x) for x in pred]
        except:
            out[var] = [float(series.iloc[-7:].mean()) for _ in range(req.horizon)]
    last_date = pd.to_datetime(df['date']).max()
    dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(req.horizon)]
    return {'dates': dates, 'forecast': out}
