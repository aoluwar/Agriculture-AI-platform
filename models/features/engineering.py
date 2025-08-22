import pandas as pd, numpy as np

def growing_degree_days(df_weather, base_c, upper_c):
    tavg = ((df_weather['tmin_c'] + df_weather['tmax_c'])/2).clip(lower=base_c, upper=upper_c)
    return (tavg - base_c).clip(lower=0.0)

def penman_monteith_et0(row):
    tmean=(row['tmin_c']+row['tmax_c'])/2.0
    wind = row.get('wind_ms', 2.0); rh = row.get('rh_mean', 60.0)
    et0 = max(0.0, 0.0023 * (tmean + 17.8) * np.sqrt(max(0.0, row['tmax_c'] - row['tmin_c'])))
    et0 *= (1.0 + 0.1*(wind-2.0)) * (1.0 + 0.002*(50.0 - rh))
    return float(et0)

def add_aggregates(weather, base_c, upper_c):
    w = weather.copy(); w['gdd'] = growing_degree_days(w, base_c, upper_c); w['et0_mm'] = w.apply(penman_monteith_et0, axis=1)
    w['cdd'] = (w['tmax_c'] > 35).astype(int); w['hdd'] = (w['tmin_c'] < 5).astype(int); return w
