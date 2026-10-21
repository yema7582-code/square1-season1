import argparse,os
_ap=argparse.ArgumentParser()
_ap.add_argument('--train',default=os.environ.get('TRAIN_CSV','./data/train.csv'))
_ap.add_argument('--test',default=os.environ.get('TEST_CSV','./data/test.csv'))
_ap.add_argument('--out',default=os.environ.get('OUT_DIR','./submissions'))
ARGS=_ap.parse_args();os.makedirs(ARGS.out,exist_ok=True)
import json,time,numpy as np,pandas as pd,lightgbm as lgb
import model as M
t=time.time();log=open('real_progress.log','w')
def P(*a): print(*a,round(time.time()-t),file=log,flush=True)
M.gbm=lambda seed: lgb.LGBMRegressor(objective='l1',n_estimators=1500,learning_rate=0.04,num_leaves=63,min_child_samples=30,subsample=.8,subsample_freq=1,colsample_bytree=.6,reg_lambda=1.0,random_state=seed,verbose=-1,n_jobs=3)
tr=pd.read_csv(ARGS.train);te=pd.read_csv(ARGS.test)
ts=pd.to_datetime(tr.datetime_utc,utc=True);cut=ts.max()-pd.DateOffset(months=12);fit,val=tr[ts<cut],tr[ts>=cut]
f2=fit.assign(m=ts[ts<cut].dt.month,s=ts[ts<cut].dt.strftime('%H%M'),cf=fit.generation_mw/fit.installed_capacity_mwp);tab=f2.groupby(['m','s']).cf.mean()
vv=val.assign(m=ts[ts>=cut].dt.month,s=ts[ts>=cut].dt.strftime('%H%M'));b1=vv.join(tab,on=['m','s']).cf*vv.installed_capacity_mwp
rep={'baseline_month_slot_cf':M.mae(b1,val.generation_mw)};P(rep)
raw=[c for c in tr.columns if c not in('id','datetime_utc','generation_mw')]
def cal(d):
    x=pd.to_datetime(d.datetime_utc);return d[raw].assign(hod=x.dt.hour+x.dt.minute/60,doy=x.dt.dayofyear,month=x.dt.month)
rep['default_gbt']=M.mae(lgb.LGBMRegressor(random_state=7,verbose=-1,n_jobs=3).fit(cal(fit),fit.generation_mw).predict(cal(val)),val.generation_mw);P(rep)
p,m,feats=M.fit_predict(fit,val,seeds=1);y=val.set_index('id').generation_mw.reindex(p.index)
rep['model']=M.mae(p,y);day=p>50;ss=np.linspace(.9,1.05,31)
sc=float(ss[int(np.argmin([np.mean(np.abs(s*p[day]-y[day])) for s in ss]))]);rep['scale']=sc
rep['model_scaled_in_sample_scale']=M.mae(p.where(p<=50,p*sc),y)
# segment errors
vt=pd.to_datetime(val.set_index('id').datetime_utc.reindex(p.index),utc=True)
rep['mae_by_month']={int(k):round(float(v),1) for k,v in (p-y).abs().groupby(vt.dt.month.values).mean().items()}
rep['top_feats']=pd.Series(m.feature_importances_,feats).sort_values(ascending=False).head(10).to_dict()
P(json.dumps(rep));json.dump(rep,open('report_real.json','w'),indent=1)
p2,_,_=M.fit_predict(tr,te,seeds=2)
S=os.path.join(ARGS.out,'')
pd.DataFrame({'id':te.id,'generation_mw':p2.where(p2<=50,p2*sc).reindex(te.id).round(2).values}).to_csv(S+'sub1_gbm_scaled.csv',index=False)
pd.DataFrame({'id':te.id,'generation_mw':p2.reindex(te.id).round(2).values}).to_csv(S+'sub2_gbm_unscaled.csv',index=False);P('wrote')
