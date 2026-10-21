"""Breathe (Square1 Ai Season 1): daily PM2.5 at 250 EPA sites.
Usage: python model.py --train train.csv --test test.csv --out submission.csv
Validation mirrors the test split: fit on dates < 2024-04-01, score Apr-Dec 2024 (same months as the
2025 test period), grouped by whole dates. No PM2.5 from the test period exists, so every site signal
comes from the training period only."""
import argparse, json, numpy as np, pandas as pd, lightgbm as lgb
SEED = 7

def clear_sky_ghi(lat, doy):
    """Daily top-of-atmosphere irradiance x 0.75 (kWh/m2/day): a cheap clear-sky reference."""
    phi = np.radians(lat); dec = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    e0 = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    h0 = 24 / np.pi * 1.367 * e0 * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return 0.75 * h0

def build(tr, te):
    al = pd.concat([tr.assign(_tr=1), te.assign(_tr=0)], ignore_index=True)
    al['date'] = pd.to_datetime(al.date)
    al['doy'] = al.date.dt.dayofyear; al['month'] = al.date.dt.month; al['dow'] = al.date.dt.dayofweek
    al['year'] = al.date.dt.year
    al['cs'] = clear_sky_ghi(al.lat_r, al.doy)
    al['kt'] = al.ghi_kwh_m2 / al.cs.clip(lower=0.05)          # haze/smoke/cloud dims this
    al['trange'] = al.temp_max_c - al.temp_min_c                  # inversions: small range, stagnant air
    al['stagnant'] = (al.wind_ms < 2.5).astype(int) * (al.precip_mm < 1).astype(int)
    # site climatology from TRAIN only
    t = al[al._tr == 1]
    sm = t.groupby(['site', 'month']).pm25.median().rename('site_month_med')
    s_all = t.groupby('site').pm25.agg(site_med='median', site_q90=lambda x: x.quantile(.9))
    al = al.join(sm, on=['site', 'month']).join(s_all, on='site')
    # same-day anomalies across neighbours (division) and per site: regional events show up together
    for c in ['kt', 'wind_ms', 'rh_pct', 'temp_c', 'precip_mm', 'pressure_kpa', 'trange']:
        clim = al.groupby(['site', 'month'])[c].transform('mean')
        al[f'{c}_anom'] = al[c] - clim
        al[f'{c}_div_anom'] = al.groupby(['division', 'date'])[f'{c}_anom'].transform('mean')
        al[f'{c}_nat_anom'] = al.groupby('date')[f'{c}_anom'].transform('mean')
    # weather lags/leads at the same site (weather is shipped for every date, so this is fair)
    al = al.sort_values(['site', 'date'])
    for c in ['wind_ms', 'precip_mm', 'kt', 'pressure_kpa']:
        g = al.groupby('site')[c]
        al[f'{c}_l1'] = g.shift(1); al[f'{c}_f1'] = g.shift(-1); al[f'{c}_r3'] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
    al['division'] = al.division.astype('category'); al['site_c'] = al.site.astype('category')
    return al

FEATS_DROP = {'id', 'pm25', '_tr', 'date', 'site'}

def fit_predict(tr, te):
    al = build(tr, te); feats = [c for c in al.columns if c not in FEATS_DROP]
    a, b = al[al._tr == 1], al[al._tr == 0]
    ps = []
    for seed in range(3):
        m = lgb.LGBMRegressor(objective='l1', n_estimators=2500, learning_rate=0.02, num_leaves=63,
                              min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                              random_state=SEED + seed, verbose=-1)
        m.fit(a[feats], a.pm25); ps.append(m.predict(b[feats]))
    return pd.Series(np.mean(ps, 0).clip(0), index=b.id.values), m, feats

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--test')
    ap.add_argument('--out', default='submission.csv'); a = ap.parse_args()
    tr = pd.read_csv(a.train); d = pd.to_datetime(tr.date)
    fit, val = tr[d < '2024-04-01'], tr[(d >= '2024-04-01') & (d <= '2024-12-31')]
    base = val.join(fit.assign(month=pd.to_datetime(fit.date).dt.month).groupby(['site', 'month']).pm25.mean().rename('b'),
                    on=['site', pd.to_datetime(val.date).dt.month.rename('month')])
    p, m, feats = fit_predict(fit, val.drop(columns='pm25'))
    rep = {'baseline_site_month_mean': float(np.nanmean(np.abs(base.b - val.pm25))),
           'model': float(np.mean(np.abs(p.reindex(val.id).values - val.pm25.values)))}
    print(json.dumps(rep, indent=1)); json.dump(rep, open('report.json', 'w'), indent=1)
    if a.test:
        te = pd.read_csv(a.test); p, _, _ = fit_predict(tr, te)
        pd.DataFrame({'id': te.id, 'pm25': p.reindex(te.id).round(3).values}).to_csv(a.out, index=False); print('wrote', a.out)
