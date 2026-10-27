from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
from typing import Any
import numpy as np

RAW_KEYS = (
    "patient_id","site_status_pre","site_status_post","site_change",
    "site_burden_pre","site_burden_post","site_size_pre","site_size_post",
    "site_relation_pre","site_relation_post","global_numeric","global_categorical"
)

def sha256(path: Path|str) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(path, fieldnames, rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

def load_jsonl(path):
    rows=[json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    if not rows: raise ValueError(f'Empty JSONL: {path}')
    return rows

def _num(x): return np.nan if x is None or x=='' else float(x)

def structured_to_raw_arrays(rows: list[dict[str,Any]], schema: dict[str,Any]) -> dict[str,np.ndarray]:
    order=list(schema['site_order']); status=dict(schema['status_vocab']); change=dict(schema['change_vocab'])
    n=len(rows); s=len(order)
    if s!=14: raise ValueError('Schema must contain 14 sites')
    out={
      'patient_id':np.empty(n,np.int32),
      'site_status_pre':np.empty((n,s),np.int16),'site_status_post':np.empty((n,s),np.int16),'site_change':np.empty((n,s),np.int16),
      'site_burden_pre':np.full((n,s),np.nan,np.float32),'site_burden_post':np.full((n,s),np.nan,np.float32),
      'site_size_pre':np.full((n,s),np.nan,np.float32),'site_size_post':np.full((n,s),np.nan,np.float32),
      'site_relation_pre':np.full((n,s),np.nan,np.float32),'site_relation_post':np.full((n,s),np.nan,np.float32),
      'global_numeric':np.full((n,12),np.nan,np.float32),'global_categorical':np.empty((n,4),np.int16),
    }
    seen=set()
    for i,r in enumerate(rows):
        pid=int(r['patient_id'])
        if pid in seen: raise ValueError(f'Duplicate patient_id {pid}')
        seen.add(pid); out['patient_id'][i]=pid
        sites=r.get('sites')
        if not isinstance(sites,list): raise ValueError(f'patient {pid}: sites must be list')
        sm={x.get('site'):x for x in sites if isinstance(x,dict)}
        if list(sm.keys())!=order:
            missing=[x for x in order if x not in sm]; extra=[x for x in sm if x not in order]
            if missing or extra: raise ValueError(f'patient {pid}: site mismatch missing={missing} extra={extra}')
        for j,site in enumerate(order):
            x=sm[site]
            for fld,vocab,target in [('pre_status',status,'site_status_pre'),('post_status',status,'site_status_post'),('radiologist_reported_change',change,'site_change')]:
                val=x.get(fld,'__MISSING__')
                if val not in vocab: raise ValueError(f'patient {pid} {site}: invalid {fld}={val}')
                out[target][i,j]=vocab[val]
            for target,fld in [('site_burden_pre','pre_burden_grade'),('site_burden_post','post_burden_grade'),('site_size_pre','pre_max_size_mm'),('site_size_post','post_max_size_mm'),('site_relation_pre','pre_relation_grade'),('site_relation_post','post_relation_grade')]:
                out[target][i,j]=_num(x.get(fld))
        op=r.get('overall_peritoneal') or {}
        pre=_num(op.get('pre_group_max_size_mm')); post=_num(op.get('post_group_max_size_mm'))
        g=out['global_numeric'][i]; g[0]=pre; g[1]=post
        if np.isfinite(pre) and np.isfinite(post):
            g[2]=post-pre
            if pre>0: g[3]=(post-pre)/pre
        fluid=r.get('abdominopelvic_fluid') or {}; pleural=r.get('pleural_effusion') or {}
        fp=fluid.get('pre') or {}; fq=fluid.get('post') or {}; pp=pleural.get('pre') or {}; pq=pleural.get('post') or {}
        g[4]=_num(fp.get('grade')); g[5]=_num(fq.get('grade')); g[6]=_num(pp.get('grade')); g[7]=_num(pq.get('grade'))
        g[8]=sum(sm[x].get('pre_status')=='present' for x in order)
        g[9]=sum(sm[x].get('post_status')=='present' for x in order)
        # This exactly matches the frozen Development392 tensor contract: definite transition only.
        g[10]=sum(sm[x].get('pre_status')=='absent' and sm[x].get('post_status')=='present' for x in order)
        g[11]=sum(sm[x].get('pre_status')=='present' and sm[x].get('post_status')=='absent' for x in order)
        cats=[]
        for obj in (fp,fq,pp,pq):
            st=obj.get('status','__MISSING__')
            if st not in status: raise ValueError(f'patient {pid}: invalid global status {st}')
            cats.append(status[st])
        out['global_categorical'][i]=cats
    return out

def save_raw_npz(path, arrays):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**{k:arrays[k] for k in RAW_KEYS})

def load_raw_npz(path):
    with np.load(path,allow_pickle=False) as z:
        if 'y' in z.files or 'r0' in z.files or 'y_engineering_only' in z.files:
            raise ValueError(f'Blind input contains a label field: {path}')
        missing=[k for k in RAW_KEYS if k not in z.files]
        if missing: raise ValueError(f'Missing raw feature arrays: {missing}')
        return {k:z[k].copy() for k in RAW_KEYS}

def raw_arrays_to_feature_batch(arrays, schema):
    from model3_core.data import build_features
    return build_features(arrays,schema)

def compare_raw_arrays(a,b):
    diffs={}
    for k in RAW_KEYS:
        x=np.asarray(a[k]); y=np.asarray(b[k])
        if x.shape!=y.shape: diffs[k]={'shape_a':list(x.shape),'shape_b':list(y.shape)}; continue
        if np.issubdtype(x.dtype,np.floating) or np.issubdtype(y.dtype,np.floating):
            same=np.isclose(x.astype(float),y.astype(float),equal_nan=True,atol=0,rtol=0)
        else: same=(x==y)
        n=int(np.size(same)-np.count_nonzero(same))
        if n: diffs[k]={'n_mismatch':n}
    return diffs

def fast_auc(y,score):
    y=np.asarray(y,dtype=np.int64); score=np.asarray(score,dtype=np.float64)
    order=np.argsort(score,kind='mergesort'); ys=y[order]; ss=score[order]
    _,first,counts=np.unique(ss,return_index=True,return_counts=True)
    ranks=np.repeat((first+1+first+counts)/2.0,counts); npos=int(ys.sum()); nneg=len(ys)-npos
    if npos==0 or nneg==0: raise ValueError('both classes required')
    return float((np.sum(ranks[ys==1])-npos*(npos+1)/2.0)/(npos*nneg))

def bootstrap_metric(y,p,metric,n_boot=5000,seed=20261021):
    rng=np.random.RandomState(seed); y=np.asarray(y); p=np.asarray(p); n=len(y); vals=[]
    for _ in range(n_boot):
        ix=rng.randint(0,n,n); yy=y[ix]
        if len(np.unique(yy))<2: continue
        vals.append(metric(yy,p[ix]))
    return float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5)),len(vals)
