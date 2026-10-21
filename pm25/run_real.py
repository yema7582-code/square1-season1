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
def fp(tr,te,seeds=1,obj='l1'):
    al=M.build(tr,te);feats=[c for c in al.columns if c not in M.FEATS_DROP];a,b=al[al._tr==1],al[al._tr==0];ps=[]
    for s in range(seeds):
        m=lgb.LGBMRegressor(objective=obj,n_estimators=1200,learning_rate=0.04,num_leaves=63,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.7,random_state=7+s,verbose=-1,n_jobs=3).fit(a[feats],a.pm25);ps.append(m.predict(b[feats]))
    return pd.Series(np.mean(ps,0).clip(0),index=b.id.values),m,feats
tr=pd.read_csv(ARGS.train);te=pd.read_csv(ARGS.test);d=pd.to_datetime(tr.date)
fit,val=tr[d<'2024-04-01'],tr[(d>='2024-04-01')&(d<='2024-12-31')]
mo=lambda x:pd.to_datetime(x.date).dt.month
tab=fit.assign(month=mo(fit)).groupby(['site','month']).pm25.mean().rename('b')
base=val.assign(month=mo(val)).join(tab,on=['site','month'])
rep={'baseline_site_month_mean':float(np.nanmean(np.abs(base.b-val.pm25)))}
raw=['lat_r','lon_r','temp_c','temp_max_c','temp_min_c','rh_pct','wind_ms','precip_mm','pressure_kpa','ghi_kwh_m2']
cal=lambda x:x[raw].assign(month=mo(x),doy=pd.to_datetime(x.date).dt.dayofyear)
rep['default_gbt']=float(np.mean(np.abs(lgb.LGBMRegressor(verbose=-1,n_jobs=3).fit(cal(fit),fit.pm25).predict(cal(val))-val.pm25)));P(rep)
y=val.set_index('id').pm25
for obj in['l1','huber']:
    p,m,feats=fp(fit,val.drop(columns='pm25'),obj=obj);rep['model_'+obj]=float(np.mean(np.abs(p-y.reindex(p.index))));P(rep)
    if obj=='l1':
        e=(p-y.reindex(p.index)).abs();yy=y.reindex(p.index)
        rep['mae_when_pm25_gt_15']=float(e[yy>15].mean());rep['share_rows_gt_15']=float((yy>15).mean())
        rep['mae_by_division']=e.groupby(val.set_index('id').division.reindex(p.index).values).mean().round(2).to_dict()
        rep['top_feats']=pd.Series(m.feature_importances_,feats).sort_values(ascending=False).head(10).astype(int).to_dict()
json.dump(rep,open('report_real.json','w'),indent=1)
S=os.path.join(ARGS.out,'')
p,_,_=fp(tr,te,seeds=2);pd.DataFrame({'id':te.id,'pm25':p.reindex(te.id).round(3).values}).to_csv(S+'sub1_l1_gbm.csv',index=False);P('wrote1')
p2,_,_=fp(tr,te,seeds=1,obj='huber');pd.DataFrame({'id':te.id,'pm25':(0.5*p+0.5*p2.reindex(p.index)).reindex(te.id).round(3).values}).to_csv(S+'sub2_l1_huber_blend.csv',index=False);P('wrote2')
