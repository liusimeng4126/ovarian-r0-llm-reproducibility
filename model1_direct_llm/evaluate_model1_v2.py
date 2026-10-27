#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,confusion_matrix,brier_score_loss

def read_pred(p):
    with open(p,encoding='utf-8-sig',newline='') as f: return {int(float(r['patient_id'])):float(r['p_r0']) for r in csv.DictReader(f)}
def read_lab(p):
    with open(p,encoding='utf-8-sig',newline='') as f: return {int(float(r['patient_id'])):r for r in csv.DictReader(f)}
def boot(y,p,n=5000,seed=20261019):
    rng=np.random.default_rng(seed); A=[]; AP=[]; B=[]; m=len(y)
    for _ in range(n):
        idx=rng.integers(0,m,m); yy=y[idx]; pp=p[idx]
        if len(np.unique(yy))<2: continue
        A.append(roc_auc_score(yy,pp)); AP.append(average_precision_score(yy,pp)); B.append(brier_score_loss(yy,pp))
    q=lambda x:[float(v) for v in np.percentile(x,[2.5,97.5])]
    return {'auroc_95ci':q(A),'auprc_95ci':q(AP),'brier_95ci':q(B),'bootstrap_valid':len(A),'seed':seed}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['engineering','shadow'],required=True); ap.add_argument('--predictions',required=True); ap.add_argument('--labels',default=r'prepared_same_split\MODEL1V2_SPLIT_AND_LABELS.csv'); ap.add_argument('--manifest',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--threshold-json'); a=ap.parse_args()
    P=read_pred(a.predictions); L=read_lab(a.labels); man=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    if man.get('prompt_sha256') != __import__('model1_v2_prompt').PROMPT_SHA256: raise ValueError('Prompt hash mismatch between frozen code and run manifest')
    if man.get('model')!='qwen3.8-max' or float(man.get('temperature'))!=0 or man.get('enable_thinking') is not False: raise ValueError('Run settings are not the frozen Model1-v2 settings')
    target='engineering' if a.phase=='engineering' else 'shadow_validation'; ids=[i for i,r in L.items() if r['model1v2_split']==target]; expected=292 if a.phase=='engineering' else 100
    if len(ids)!=expected: raise ValueError(f'Expected {expected} labels, got {len(ids)}')
    miss=[i for i in ids if i not in P]
    if miss: raise ValueError(f'Missing predictions: {miss[:20]}')
    y=np.array([int(L[i]['r0']) for i in ids]); p=np.array([P[i] for i in ids]); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    auroc=float(roc_auc_score(y,p)); auprc=float(average_precision_score(y,p)); brier=float(brier_score_loss(y,p))
    if a.phase=='engineering':
        fpr,tpr,thr=roc_curve(y,p); j=tpr-fpr; k=int(np.nanargmax(j)); threshold=float(thr[k])
        frozen={'model':'Model1-v2 clinically instructed direct LLM','threshold_p_r0':threshold,'selection':'Youden J on Engineering292 only','n':len(ids),'prompt_sha256':man['prompt_sha256'],'prediction_sha256':hashlib.sha256(Path(a.predictions).read_bytes()).hexdigest()}
        (out/'MODEL1_V2_THRESHOLD_FROZEN.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    else:
        if not a.threshold_json: raise ValueError('--threshold-json required for shadow')
        frozen=json.loads(Path(a.threshold_json).read_text(encoding='utf-8')); threshold=float(frozen['threshold_p_r0'])
        if frozen['prompt_sha256']!=man['prompt_sha256']: raise ValueError('Threshold prompt hash != shadow prompt hash')
    yh=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel()
    r={'phase':a.phase,'n':len(ids),'r0':int(y.sum()),'non_r0':int((1-y).sum()),'r0_prevalence':float(y.mean()),'auroc_r0':auroc,'auprc_r0':auprc,'brier':brier,'threshold_p_r0':threshold,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'sensitivity_R0':float(tp/(tp+fn)) if tp+fn else None,'specificity':float(tn/(tn+fp)) if tn+fp else None,'ppv':float(tp/(tp+fp)) if tp+fp else None,'npv':float(tn/(tn+fn)) if tn+fn else None,'accuracy':float((tp+tn)/len(y)),'prompt_sha256':man['prompt_sha256'],'prediction_sha256':hashlib.sha256(Path(a.predictions).read_bytes()).hexdigest(),'bootstrap':boot(y,p)}
    (out/f'MODEL1_V2_{a.phase.upper()}_METRICS.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
