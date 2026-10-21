import argparse,os
_ap=argparse.ArgumentParser()
_ap.add_argument('--train',default=os.environ.get('TRAIN_CSV','./data/train.csv'))
_ap.add_argument('--test',default=os.environ.get('TEST_CSV','./data/test.csv'))
_ap.add_argument('--out',default=os.environ.get('OUT_DIR','./submissions'))
ARGS=_ap.parse_args();os.makedirs(ARGS.out,exist_ok=True)
import json,time,numpy as np,pandas as pd,lightgbm as lgb

class QR:
    def fit(self,X,y,it=60):
        X=np.c_[np.ones(len(X)),X];b=np.linalg.lstsq(X,y,rcond=None)[0]
        for _ in range(it):
            w=1/np.maximum(np.abs(y-X@b),1e-3);b=np.linalg.solve((X*w[:,None]).T@X,(X*w[:,None]).T@y)
        self.b=b;return self
    def predict(self,X): return np.c_[np.ones(len(X)),X]@self.b
import model as M
t=time.time();log=open('real_progress.log','a')
def P(*a): print(*a,round(time.time()-t),file=log,flush=True)
tr=pd.read_csv(ARGS.train);te=pd.read_csv(ARGS.test)
def fp(tr,te,ord_feats=True):
    Xtr,Xte,y=M.features(tr),M.features(te),tr.home_margin.values
    if not ord_feats:
        drop=[c for c in Xtr if c in('home_g','away_g','season_day') or 'mom' in c or 'start' in c]
        Xtr,Xte=Xtr.drop(columns=drop),Xte.drop(columns=drop)
    w=0.5**((tr.season.max()-tr.season.values)/12); rec=tr.season.values>=tr.season.max()-14
    cols=['logit','neutral','playoffs','home_b2b','away_b2b']
    q=QR().fit(Xtr[cols].fillna(0).values[rec],y[rec])
    pq=q.predict(Xte[cols].fillna(0).values)
    g=lgb.LGBMRegressor(objective='l1',n_estimators=900,learning_rate=0.02,num_leaves=15,min_child_samples=80,subsample=.8,subsample_freq=1,colsample_bytree=.8,random_state=7,verbose=-1,n_jobs=2).fit(Xtr,y,sample_weight=w)
    pg=g.predict(Xte); return {'qr':np.asarray(pq),'gbm':pg,'blend':.5*np.asarray(pq)+.5*pg}
last=tr.season.max();vm=tr.season>last-3;fit,val=tr[~vm].reset_index(drop=True),tr[vm].reset_index(drop=True)
rep={'home+3':M.mae(np.where(val.neutral==1,0,3),val.home_margin)}
for k,o in [('A_past_momentum',True),('B_elo_only',False)]:
    for kk,v in fp(fit,val,o).items(): rep[f'{k}_{kk}']=M.mae(v,val.home_margin)
    P(json.dumps(rep))
json.dump(rep,open('report_real.json','w'),indent=1)
S=os.path.join(ARGS.out,'')
for name,o in [('sub1_blend_past_momentum.csv',True),('sub2_blend_elo_only.csv',False)]:
    p=fp(tr,te,o)['blend']; pd.DataFrame({'id':te.id,'home_margin':np.round(p,3)}).to_csv(S+name,index=False); P('wrote',name)
