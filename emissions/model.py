"""The emissions gap (Square1 Ai Season 1): annual CO2e per fossil power plant-year.
Usage: python model.py --train train.csv --test test.csv --out submission.csv
Validation mirrors the split: GroupKFold over country x 0.5-degree cells (whole cells held out).
Target: log1p(tonnes), RMSE on it == the competition's RMSLE."""
import argparse, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
SEED = 7
FUELS = ['coal', 'gas', 'oil', 'other_fossil', 'biomass', 'waste']

def build(tr, te):
    al = pd.concat([tr.assign(_tr=1), te.assign(_tr=0)], ignore_index=True)
    al['cell'] = al.country + '_' + al.lat_r.astype(str) + '_' + al.lon_r.astype(str)
    al['logcap'] = np.log1p(al.capacity_mw)
    for f in FUELS: al[f'has_{f}'] = al.fuels.fillna('').str.contains(f).astype(int)
    # plant-level descriptors (features only, no target): how its capacity moves across years
    g = al.groupby('plant')
    al['n_years'] = g.year.transform('count'); al['cap_max'] = g.capacity_mw.transform('max')
    al['cap_ratio'] = al.capacity_mw / al.cap_max.clip(lower=1e-3)
    al['first_year'] = g.year.transform('min')
    # cell / country context, features only
    al['cell_plants'] = al.groupby(['cell', 'year']).plant.transform('nunique')
    al['cell_cap'] = np.log1p(al.groupby(['cell', 'year']).capacity_mw.transform('sum'))
    al['country_cap'] = np.log1p(al.groupby(['country', 'year']).capacity_mw.transform('sum'))
    al['cap_rank_country'] = al.groupby(['country', 'year']).capacity_mw.rank(pct=True)
    for c in ['country', 'primary_fuel', 'fuels']: al[c] = al[c].astype('category')
    al['y'] = np.log1p(al.annual_co2e_t)
    return al

def target_enc(a, b, keys, name, k=20):
    """Smoothed mean of log tonnes-per-MW, computed on rows `a` only."""
    a = a.copy(); a['ti'] = a.y - a.logcap; prior = a.ti.mean()
    s = a.groupby(keys, observed=True).ti.agg(['sum', 'count'])
    enc = (s['sum'] + k * prior) / (s['count'] + k)
    return b.join(enc.rename(name), on=keys)[name].fillna(prior).values

TE_KEYS = {'te_cf': ['country', 'primary_fuel'], 'te_f': ['primary_fuel'], 'te_fy': ['primary_fuel', 'year'], 'te_c': ['country']}

def add_te(a, b):
    b = b.copy()
    for n, k in TE_KEYS.items(): b[n] = target_enc(a, b, k, n)
    return b

def model():
    return lgb.LGBMRegressor(objective='regression', n_estimators=1500, learning_rate=0.02, num_leaves=31,
                             min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                             reg_lambda=2.0, random_state=SEED, verbose=-1)

DROP = {'id', 'plant', 'annual_co2e_t', '_tr', 'y', 'cell'}  # 'base' stays in as a feature

def oof_te(a, groups, n=5):
    """Out-of-fold target encodings for the training rows (no row sees its own cell's target)."""
    out = a.copy()
    for n_ in TE_KEYS: out[n_] = np.nan
    for i, j in GroupKFold(n).split(a, groups=groups):
        enc = add_te(a.iloc[i], a.iloc[j])
        for n_ in TE_KEYS: out.iloc[j, out.columns.get_loc(n_)] = enc[n_].values
    return out

def fit_predict(a, b):
    a2 = oof_te(a, a.cell); b2 = add_te(a, b)
    for d in (a2, b2): d['base'] = np.log1p(np.expm1(d.logcap) * np.exp(d.te_cf))  # cap x tonnes-per-MW prior
    feats = [c for c in a2.columns if c not in DROP]
    m = model().fit(a2[feats], a2.y - a2.base)   # learn the residual on top of the physical prior
    return np.clip(np.expm1(b2.base.values + m.predict(b2[feats])), 0, None), m, feats

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--test')
    ap.add_argument('--out', default='submission.csv'); a = ap.parse_args()
    tr = pd.read_csv(a.train); te = pd.read_csv(a.test) if a.test else tr.iloc[:0].drop(columns='annual_co2e_t')
    al = build(tr, te); A = al[al._tr == 1].reset_index(drop=True)
    rm = lambda p, y: float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2)))
    sc, sb = [], []
    for i, j in GroupKFold(5).split(A, groups=A.cell):
        p, _, _ = fit_predict(A.iloc[i].reset_index(drop=True), A.iloc[j].reset_index(drop=True)); sc.append(rm(p, A.annual_co2e_t.iloc[j]))
        med = (A.iloc[i].annual_co2e_t / A.iloc[i].capacity_mw).groupby(A.iloc[i].primary_fuel, observed=True).median()
        sb.append(rm(A.iloc[j].capacity_mw * A.iloc[j].primary_fuel.map(med).astype(float), A.annual_co2e_t.iloc[j]))
    rep = {'cv_rmsle_model': float(np.mean(sc)), 'cv_rmsle_baseline_cap_x_median': float(np.mean(sb))}
    print(json.dumps(rep, indent=1)); json.dump(rep, open('report.json', 'w'), indent=1)
    if a.test:
        B = al[al._tr == 0].reset_index(drop=True); p, _, _ = fit_predict(A, B)
        pd.DataFrame({'id': B.id, 'annual_co2e_t': np.round(p, 2)}).to_csv(a.out, index=False); print('wrote', a.out)
