"""Estimate the sun (Square1 Ai Season 1): GB national solar generation per half-hour.
Usage: python model.py --train train.csv [--test test.csv --out submission.csv]
Validation mirrors the test split: fit Jan 2022 - Jun 2024, score Jul 2024 - Jun 2025 (a full
following year, like the real test year). Target is modelled as capacity factor, weighted by capacity,
so the L1 loss equals MAE in MW."""
import argparse, json, warnings, numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
PTS = ['london', 'birmingham', 'manchester', 'leeds', 'bristol', 'norwich', 'southampton', 'plymouth',
       'cardiff', 'newcastle', 'glasgow', 'edinburgh']
LATLON = {'london': (51.507, -0.128), 'birmingham': (52.486, -1.890), 'manchester': (53.481, -2.242),
          'leeds': (53.800, -1.549), 'bristol': (51.454, -2.588), 'norwich': (52.630, 1.297),
          'southampton': (50.910, -1.404), 'plymouth': (50.376, -4.143), 'cardiff': (51.481, -3.179),
          'newcastle': (54.978, -1.618), 'glasgow': (55.864, -4.252), 'edinburgh': (55.953, -3.188)}
# rough share of GB PV capacity near each point (south-heavy); used only for a weighted average
WT = dict(london=.12, birmingham=.10, manchester=.06, leeds=.06, bristol=.12, norwich=.11, southampton=.14,
          plymouth=.10, cardiff=.08, newcastle=.04, glasgow=.035, edinburgh=.035)
SEED = 7

def sun(ts, lat, lon):
    """Solar elevation (deg) at the middle of the half-hour (NOAA-style approximation)."""
    t = ts + pd.Timedelta(minutes=15)
    doy = t.dt.dayofyear.values; hr = t.dt.hour.values + t.dt.minute.values / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = 0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
    ha = np.radians((hr * 60 + eqt + 4 * lon) / 4 - 180)
    phi = np.radians(lat)
    cz = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(cz, -1, 1))), np.degrees(ha)

def build(df):
    df = df.copy(); ts = pd.to_datetime(df.datetime_utc, utc=True); df['_ts'] = ts
    df = df.sort_values('_ts').reset_index(drop=True); ts = df._ts
    X = pd.DataFrame(index=df.index)
    X['doy'] = ts.dt.dayofyear; X['hod'] = ts.dt.hour + ts.dt.minute / 60; X['half'] = (ts.dt.minute == 30).astype(int)
    X['t_index'] = (ts - pd.Timestamp('2022-01-01', tz='UTC')).dt.days / 365.25
    el, ha = sun(ts, 52.6, -1.5); X['sun_el'] = el; X['sun_ha'] = ha
    X['sun_el_prev'], _ = sun(ts - pd.Timedelta(minutes=30), 52.6, -1.5)
    X['sun_el_next'], _ = sun(ts + pd.Timedelta(minutes=30), 52.6, -1.5)
    # contiguous hourly series: the hour before/after for every point (features only; rows are sorted by time)
    hourly_key = ts.dt.floor('h')
    kt_all, ghi_w, ghi_all = [], 0, []
    for p in PTS:
        lat, lon = LATLON[p]
        e, _ = sun(ts, lat, lon); X[f'{p}_el'] = e
        g, c = df[f'{p}_ghi'], df[f'{p}_clearsky']
        H = pd.DataFrame({'h': hourly_key, 'g': g, 'c': c}).groupby('h').first()
        prev = H.g.shift(1, freq='h').reindex(hourly_key).values; nxt = H.g.shift(-1, freq='h').reindex(hourly_key).values
        X[f'{p}_ghi'] = g; X[f'{p}_cs'] = c
        # interpolate hourly GHI toward the half-hour's own time: first half leans on the previous hour, second on the next
        X[f'{p}_ghi_i'] = np.where(X.half == 0, 0.75 * g + 0.25 * np.nan_to_num(prev, nan=g), 0.75 * g + 0.25 * np.nan_to_num(nxt, nan=g))
        X[f'{p}_ghi_prev'] = prev; X[f'{p}_ghi_next'] = nxt
        X[f'{p}_kt'] = (g / c.where(c > 5)).clip(0, 1.3)
        for v in ['temp_c', 'cloud_pct', 'wind_ms', 'rh_pct']: X[f'{p}_{v}'] = df[f'{p}_{v}']
        kt_all.append(X[f'{p}_kt']); ghi_all.append(X[f'{p}_ghi_i']); ghi_w = ghi_w + WT[p] * X[f'{p}_ghi_i']
    X['ghi_w'] = ghi_w; X['ghi_mean'] = np.mean(ghi_all, 0); X['ghi_std'] = np.std(ghi_all, 0)
    K = pd.concat(kt_all, axis=1); X['kt_mean'] = K.mean(1); X['kt_min'] = K.min(1); X['kt_max'] = K.max(1)
    X['cs_w'] = sum(WT[p] * df[f'{p}_clearsky'] for p in PTS)
    X['temp_mean'] = np.mean([df[f'{p}_temp_c'] for p in PTS], 0)   # hot panels lose efficiency
    X['cloud_mean'] = np.mean([df[f'{p}_cloud_pct'] for p in PTS], 0)
    X['cap'] = df.installed_capacity_mwp
    return df, X

def gbm(seed):
    return lgb.LGBMRegressor(objective='l1', n_estimators=3000, learning_rate=0.02, num_leaves=63, min_child_samples=30,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, random_state=seed, verbose=-1)

def fit_predict(tr, te, seeds=3):
    a, Xa = build(tr); b, Xb = build(te)
    cf = a.generation_mw / a.installed_capacity_mwp
    feats = [c for c in Xa.columns if c not in ('cap', 't_index', 'doy')]  # ablation: dropping the calendar trend helps (126 vs 129 MW) and avoids extrapolating into the test year
    p = np.zeros(len(b))
    for s in range(seeds):
        m = gbm(SEED + s).fit(Xa[feats], cf, sample_weight=a.installed_capacity_mwp)
        p += m.predict(Xb[feats]) / seeds
    mw = np.clip(p, 0, None) * b.installed_capacity_mwp.values
    mw[(Xb.sun_el.values < -6) & (Xb.cs_w.values <= 0)] = 0.0       # night
    return pd.Series(mw, index=b.id.values), m, feats

def year_ahead_scale(tr, seeds=1):
    """Fit on all but the last 12 months, predict those 12 months, and learn one multiplicative correction
    for daytime output (a year-ahead model over-predicts slightly; PV_Live capacity grows faster than yield)."""
    ts = pd.to_datetime(tr.datetime_utc, utc=True); cut = ts.max() - pd.DateOffset(months=12)
    p, _, _ = fit_predict(tr[ts < cut], tr[ts >= cut], seeds=seeds)
    y = tr.set_index('id').generation_mw.reindex(p.index); day = p > 50
    ss = np.linspace(0.90, 1.05, 31)
    return float(ss[int(np.argmin([np.mean(np.abs(s * p[day] - y[day])) for s in ss]))])

def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--train', default='proxy_train.csv'); ap.add_argument('--test')
    ap.add_argument('--out', default='submission.csv'); ap.add_argument('--seeds', type=int, default=3); a = ap.parse_args()
    tr = pd.read_csv(a.train); ts = pd.to_datetime(tr.datetime_utc, utc=True)
    cut = ts.max() - pd.DateOffset(months=12)  # last 12 months of train = validation year (2024-07..2025-06 on the real file)
    fit, val = tr[ts < cut], tr[ts >= cut]
    # baseline 1: train-period mean capacity factor per month x half-hour slot, times capacity
    f2 = fit.assign(m=pd.to_datetime(fit.datetime_utc).dt.month, s=pd.to_datetime(fit.datetime_utc).dt.strftime('%H%M'),
                    cf=fit.generation_mw / fit.installed_capacity_mwp)
    tab = f2.groupby(['m', 's']).cf.mean()
    vv = val.assign(m=pd.to_datetime(val.datetime_utc).dt.month, s=pd.to_datetime(val.datetime_utc).dt.strftime('%H%M'))
    b1 = vv.join(tab, on=['m', 's']).cf * vv.installed_capacity_mwp
    raw = [c for c in tr.columns if c not in ('id', 'datetime_utc', 'generation_mw')]
    def cal(d):
        t = pd.to_datetime(d.datetime_utc); return d[raw].assign(hod=t.dt.hour + t.dt.minute / 60, doy=t.dt.dayofyear, month=t.dt.month)
    dflt = lgb.LGBMRegressor(random_state=SEED, verbose=-1).fit(cal(fit), fit.generation_mw)
    p, m, feats = fit_predict(fit, val, seeds=a.seeds)
    rep = {'baseline_month_slot_cf': mae(b1, val.generation_mw), 'default_gbt': mae(dflt.predict(cal(val)), val.generation_mw),
           'model': mae(p.reindex(val.id).values, val.generation_mw.values)}
    print(json.dumps(rep, indent=1)); json.dump(rep, open('report.json', 'w'), indent=1)
    imp = pd.Series(m.feature_importances_, feats).sort_values(ascending=False); print(imp.head(15))
    if a.test:
        te = pd.read_csv(a.test); p, _, _ = fit_predict(tr, te, seeds=a.seeds)
        sc = year_ahead_scale(tr); print('year-ahead scale', sc)
        p = p.where(p <= 50, p * sc)
        pd.DataFrame({'id': te.id, 'generation_mw': p.reindex(te.id).round(2).values}).to_csv(a.out, index=False); print('wrote', a.out, len(te))
