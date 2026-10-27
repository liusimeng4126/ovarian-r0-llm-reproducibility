from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from model3_test_common import fast_auc, bootstrap_metric, write_csv, write_json, sha256

def labels_csv(path):
 d=pd.read_csv(path,encoding='utf-8-sig'); cols=list(d.columns); pid='patient_id' if 'patient_id' in cols else cols[0]; lab=next((x for x in ('r0','R0','结果','label') if x in cols),None)
 if not lab: raise ValueError(f'Internal label columns not understood: {cols}')
 return norm(d,pid,lab)
def labels_xlsx(path):
 d=pd.read_excel(path); cols=list(d.columns); pid='patient_id' if 'patient_id' in cols else cols[0]; lab=next((x for x in ('r0','R0','结果','label') if x in cols),None)
 if not lab: raise ValueError(f'External label columns not understood: {cols}')
 return norm(d,pid,lab)
def norm(d,pid,lab):
 def cv(x):
  if isinstance(x,(int,np.integer,float,np.floating)) and not pd.isna(x): return int(float(x))
  s=str(x).strip().lower().replace(' ','')
  if s in {'r0','1','是','complete','完全切除'}: return 1
  if s in {'非r0','non-r0','nonr0','0','否'}: return 0
  raise ValueError(f'Unrecognized outcome value: {x!r}')
 out=pd.DataFrame({'patient_id':d[pid].astype(int),'r0':[cv(x) for x in d[lab]]})
 if out.patient_id.duplicated().any(): raise ValueError('Duplicate outcome patient IDs')
 return out
def read_pred(path):
 d=pd.read_csv(path,encoding='utf-8-sig');
 if set(['patient_id','p_r0'])-set(d.columns): raise ValueError(f'Prediction columns invalid: {path}')
 return d
def ap(y,p):
 order=np.argsort(-p,kind='mergesort'); yy=np.asarray(y)[order]; tp=np.cumsum(yy); fp=np.cumsum(1-yy); prec=tp/(tp+fp); return float(np.sum(prec*yy)/max(1,np.sum(yy)))
def metrics(name,pred,lab,threshold):
 m=pred.merge(lab,on='patient_id',how='inner',validate='one_to_one')
 if len(m)!=len(pred) or len(m)!=len(lab): raise ValueError(f'{name} patient mismatch pred={len(pred)} labels={len(lab)} merged={len(m)}')
 y=m.r0.to_numpy(int); p=m.p_r0.to_numpy(float); auc=fast_auc(y,p); aupr=ap(y,p); ci=bootstrap_metric(y,p,fast_auc); ci_ap=bootstrap_metric(y,p,ap)
 pr=p>=threshold; tp=int(np.sum(pr&(y==1))); tn=int(np.sum((~pr)&(y==0))); fp=int(np.sum(pr&(y==0))); fn=int(np.sum((~pr)&(y==1)))
 return m,{'cohort':name,'n':len(y),'r0':int(y.sum()),'non_r0':int(len(y)-y.sum()),'auroc':auc,'auroc_ci_low':ci[0],'auroc_ci_high':ci[1],'auprc':aupr,'auprc_ci_low':ci_ap[0],'auprc_ci_high':ci_ap[1],'brier':float(np.mean((p-y)**2)),'threshold':threshold,'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'accuracy':(tp+tn)/len(y),'tp':tp,'tn':tn,'fp':fp,'fn':fn}
def paired_delta(y,pfull,palt,nboot=5000,seed=20261021):
 rng=np.random.RandomState(seed); vals=[]; y=np.asarray(y); pfull=np.asarray(pfull); palt=np.asarray(palt)
 for _ in range(nboot):
  ix=rng.randint(0,len(y),len(y)); yy=y[ix]
  if len(np.unique(yy))<2: continue
  vals.append(fast_auc(yy,palt[ix])-fast_auc(yy,pfull[ix]))
 return float(fast_auc(y,palt)-fast_auc(y,pfull)),float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))
def plots(out, internal, external, ext_abl, return_dir):
 import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
 from sklearn.metrics import roc_curve, precision_recall_curve
 from sklearn.calibration import calibration_curve
 for name,m in [('Internal100',internal),('External80',external)]:
  y=m.r0.to_numpy(); p=m.p_r0.to_numpy(); fpr,tpr,_=roc_curve(y,p); fig,ax=plt.subplots(); ax.plot(fpr,tpr,label=f'AUROC={fast_auc(y,p):.3f}'); ax.plot([0,1],[0,1],'--'); ax.set(xlabel='1 - Specificity',ylabel='Sensitivity',title=f'Model3 Full ROC - {name}'); ax.legend(); fig.tight_layout(); fig.savefig(out/f'FIG_MODEL3_{name}_ROC.png',dpi=180); plt.close(fig)
  prec,rec,_=precision_recall_curve(y,p); fig,ax=plt.subplots(); ax.plot(rec,prec); ax.set(xlabel='Recall',ylabel='Precision',title=f'Model3 Full PR - {name}'); fig.tight_layout(); fig.savefig(out/f'FIG_MODEL3_{name}_PR.png',dpi=180); plt.close(fig)
  frac,meanp=calibration_curve(y,p,n_bins=5,strategy='quantile'); fig,ax=plt.subplots(); ax.plot(meanp,frac,marker='o'); ax.plot([0,1],[0,1],'--'); ax.set(xlabel='Mean predicted p(R0)',ylabel='Observed R0 fraction',title=f'Model3 Full calibration - {name}'); fig.tight_layout(); fig.savefig(out/f'FIG_MODEL3_{name}_CALIBRATION.png',dpi=180); plt.close(fig)
 # External ablation AUROC comparison.
 names=list(ext_abl); vals=[fast_auc(ext_abl[k].r0.to_numpy(),ext_abl[k].p_r0.to_numpy()) for k in names]; fig,ax=plt.subplots(figsize=(8,4)); ax.bar(names,vals); ax.set_ylim(0,1); ax.set_ylabel('AUROC'); ax.set_title('External80 frozen Full and prespecified ablations'); fig.tight_layout(); fig.savefig(out/'FIG_MODEL3_EXTERNAL80_ABLATION_AUROC.png',dpi=180); plt.close(fig)
 # Full external site-occlusion is safer to interpret than attention alone.
 occp=Path(return_dir)/'02_EXTERNAL80_FULL'/'MODEL3_EXTERNAL80_FULL_SITE_OCCLUSION_DELTA_P.csv'
 if occp.is_file():
  occ=pd.read_csv(occp,encoding='utf-8-sig').merge(external[['patient_id','r0']],on='patient_id',validate='one_to_one'); sites=[c for c in occ.columns if c not in {'patient_id','r0'}]; means=[float(occ[c].mean()) for c in sites]; order=np.argsort(np.abs(means))[::-1]; fig,ax=plt.subplots(figsize=(9,6)); ax.barh([sites[i] for i in order],[means[i] for i in order]); ax.set(xlabel='Mean delta p(R0): Full - site-occluded',title='External80 Full site-occlusion sensitivity'); fig.tight_layout(); fig.savefig(out/'FIG_MODEL3_EXTERNAL80_SITE_OCCLUSION.png',dpi=180); plt.close(fig)
 qcp=Path(return_dir)/'04_INPUT_SHIFT_QC'/'NUMERIC_MISSINGNESS_SHIFT.csv'
 if qcp.is_file():
  q=pd.read_csv(qcp,encoding='utf-8-sig'); pv=q.pivot_table(index='feature',columns='cohort',values='missing_rate',aggfunc='mean'); fig,ax=plt.subplots(figsize=(9,6)); im=ax.imshow(pv.fillna(0).to_numpy(),aspect='auto',vmin=0,vmax=1); ax.set_yticks(range(len(pv.index)),pv.index); ax.set_xticks(range(len(pv.columns)),pv.columns,rotation=20); ax.set_title('Outcome-blind numeric missingness shift'); fig.colorbar(im,ax=ax,label='Missing rate'); fig.tight_layout(); fig.savefig(out/'FIG_MODEL3_INPUT_MISSINGNESS_SHIFT.png',dpi=180); plt.close(fig)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--return-dir',required=True); ap.add_argument('--internal-labels',required=True); ap.add_argument('--external-labels',required=True); ap.add_argument('--outdir',required=True); args=ap.parse_args()
 r=Path(args.return_dir); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True); threshold=0.3453282415866852
 il=labels_csv(args.internal_labels); el=labels_xlsx(args.external_labels); ip=read_pred(r/'01_INTERNAL100_FULL'/'MODEL3_INTERNAL100_FULL_BLIND_PREDICTIONS.csv'); ep=read_pred(r/'02_EXTERNAL80_FULL'/'MODEL3_EXTERNAL80_FULL_BLIND_PREDICTIONS.csv')
 im,ims=metrics('Internal100',ip,il,threshold); em,ems=metrics('External80',ep,el,threshold); write_csv(out/'TABLE_MODEL3_PRIMARY_PERFORMANCE.csv',list(ims),[ims,ems]); im.to_csv(out/'PATIENT_LEVEL_INTERNAL100.csv',index=False,encoding='utf-8-sig'); em.to_csv(out/'PATIENT_LEVEL_EXTERNAL80.csv',index=False,encoding='utf-8-sig')
 ext={'Full':em}; rows=[]; paired=[]
 for v in ('noSiteID','noDelta','PostOnly','noAttention'):
  p=read_pred(r/'03_EXTERNAL80_ABLATION_PREDICTIONS'/f'MODEL3_EXTERNAL80_{v}_BLIND_PREDICTIONS.csv'); m,s=metrics('External80_'+v,p,el,threshold); ext[v]=m; rows.append({'variant':v,**s}); joined=em[['patient_id','r0','p_r0']].rename(columns={'p_r0':'full'}).merge(m[['patient_id','p_r0']],on='patient_id').rename(columns={'p_r0':'alt'}); d=paired_delta(joined.r0,joined.full,joined.alt); paired.append({'variant':v,'delta_auroc_alt_minus_full':d[0],'ci95_low':d[1],'ci95_high':d[2]})
 write_csv(out/'TABLE_EXTERNAL80_ABLATIONS.csv',list(rows[0]),rows); write_csv(out/'TABLE_EXTERNAL80_PAIRED_DELTA_VS_FULL.csv',list(paired[0]),paired); plots(out,im,em,ext,r)
 write_json(out/'MODEL3_FINAL_EVALUATION_MANIFEST.json',{'threshold_frozen_from_dev392':threshold,'internal_prediction_sha256':sha256(r/'01_INTERNAL100_FULL'/'MODEL3_INTERNAL100_FULL_BLIND_PREDICTIONS.csv'),'external_prediction_sha256':sha256(r/'02_EXTERNAL80_FULL'/'MODEL3_EXTERNAL80_FULL_BLIND_PREDICTIONS.csv'),'internal_labels_sha256':sha256(args.internal_labels),'external_labels_sha256':sha256(args.external_labels),'bootstrap_n':5000,'bootstrap_seed':20261021,'ablations_secondary_only':True})
 print('='*80); print('PASS - OWNER-SIDE MODEL3 EVALUATION COMPLETE'); print('Internal100 Full AUROC',ims['auroc']); print('External80 Full AUROC',ems['auroc']); print('Saved:',out); print('='*80)
if __name__=='__main__': main()
