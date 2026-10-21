"""Call the game (Square1 Ai Season 1): home-margin model.
Usage:  python model.py --train train.csv [--test test.csv --out submission.csv]
Validation mirrors the test split: fit on seasons <= S-3, score the last three train seasons.
Only pre-game information is used. Features that look at a team's *later* games are never built
(see LEAKAGE.md)."""
import argparse, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import QuantileRegressor

SEED = 7

def past_form(df):
    """Elo momentum from each team's *earlier* games in the same season (pre-game Elo only)."""
    long = pd.concat([
        df[['id', 'season', 'home_team', 'home_seasongame', 'home_elo']].set_axis(['id', 'season', 'team', 'g', 'elo'], axis=1),
        df[['id', 'season', 'away_team', 'away_seasongame', 'away_elo']].set_axis(['id', 'season', 'team', 'g', 'elo'], axis=1).assign(side='a'),
    ]).fillna({'side': 'h'})
    long = long[long.g > 0].sort_values(['season', 'team', 'g'])
    grp = long.groupby(['season', 'team']).elo
    long['mom5'] = long.elo - grp.shift(5)          # Elo now minus Elo five games ago
    long['mom10'] = long.elo - grp.shift(10)
    long['start'] = long.elo - grp.transform('first')  # change since the season opener (all earlier games)
    out = {}
    for side in 'ha':
        s = long[long.side == side].set_index('id')[['mom5', 'mom10', 'start']]
        out[side] = s.add_prefix(f'{side}_')
    return out['h'], out['a']

def features(df):
    X = pd.DataFrame(index=df.index)
    p = df.home_forecast.clip(1e-4, 1 - 1e-4)
    X['logit'] = np.log(p / (1 - p))
    X['elo_diff'] = df.home_elo - df.away_elo
    X['neutral'] = df.neutral; X['playoffs'] = df.playoffs
    X['aba'] = (df.league == 'ABA').astype(int)
    X['season'] = df.season; X['season_day'] = df.season_day
    for s in ['home', 'away']:
        r = df[f'{s}_rest']
        X[f'{s}_rest'] = r.where(r >= 0, np.nan).clip(upper=10)
        X[f'{s}_b2b'] = (r == 1).astype(int)
    X['rest_diff'] = X.home_rest - X.away_rest
    X['home_g'] = df.home_seasongame; X['away_g'] = df.away_seasongame
    X['elo_sum'] = df.home_elo + df.away_elo
    h, a = past_form(df)
    X = X.join(h, on=df.id) if False else X
    X[h.columns] = h.reindex(df.id).values; X[a.columns] = a.reindex(df.id).values
    X['mom5_diff'] = X.h_mom5 - X.a_mom5; X['start_diff'] = X.h_start - X.a_start
    return X

def fit_predict(tr, te, weights_halflife=12):
    Xtr, Xte, y = features(tr), features(te), tr.home_margin.values
    # recency weights: the game changes (3-pt line, pace), recent seasons matter more
    w = 0.5 ** ((tr.season.max() - tr.season.values) / weights_halflife)
    preds = {}
    # 1) median regression on the 538 logit, per-neutral offset
    cols = ['logit', 'neutral', 'playoffs', 'home_b2b', 'away_b2b']
    qr_tr = Xtr[cols].fillna(0); qr_te = Xte[cols].fillna(0)
    rec = tr.season.values >= tr.season.max() - 14
    qr = QuantileRegressor(quantile=0.5, alpha=0.0, solver='highs').fit(qr_tr[rec], y[rec], sample_weight=w[rec])
    preds['qr'] = qr.predict(qr_te)
    # 2) LightGBM with L1 loss on all features
    feats = [c for c in Xtr.columns]
    g = lgb.LGBMRegressor(objective='l1', n_estimators=900, learning_rate=0.02, num_leaves=15,
                          min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          reg_lambda=1.0, random_state=SEED, verbose=-1)
    g.fit(Xtr[feats], y, sample_weight=w)
    preds['gbm'] = g.predict(Xte[feats])
    preds['blend'] = 0.5 * preds['qr'] + 0.5 * preds['gbm']
    return preds

def baselines(te):
    return {'home+3': np.where(te.neutral == 1, 0, 3)}

def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--train', default='proxy_train.csv')
    ap.add_argument('--test'); ap.add_argument('--out', default='submission.csv'); a = ap.parse_args()
    tr = pd.read_csv(a.train)
    last = tr.season.max(); val_mask = tr.season > last - 3
    fit, val = tr[~val_mask].reset_index(drop=True), tr[val_mask].reset_index(drop=True)
    rep = {}
    for k, v in baselines(val).items(): rep[k] = mae(v, val.home_margin)
    dflt = lgb.LGBMRegressor(random_state=SEED, verbose=-1)
    raw = ['season', 'playoffs', 'neutral', 'season_day', 'home_elo', 'away_elo', 'home_forecast', 'home_seasongame', 'away_seasongame', 'home_rest', 'away_rest']
    dflt.fit(fit[raw], fit.home_margin); rep['default_gbt'] = mae(dflt.predict(val[raw]), val.home_margin)
    for k, v in fit_predict(fit, val).items(): rep[k] = mae(v, val.home_margin)
    print(json.dumps(rep, indent=1)); json.dump(rep, open('report.json', 'w'), indent=1)
    if a.test:
        te = pd.read_csv(a.test)
        p = fit_predict(tr, te)['blend']
        pd.DataFrame({'id': te.id, 'home_margin': np.round(p, 3)}).to_csv(a.out, index=False)
        print('wrote', a.out, len(te))
