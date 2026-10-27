#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RRS-v3.3 BOTTLENECK / DEEP-LOGIC ENGINE
----------------------------------------
Purpose:
  Use ONLY the already-QC'd V6.1 14-site structured JSONL.
  No new Qwen/LLM call. No surgery/pathology predictors.
  Replace simple additive site weights with a clinically constrained hierarchy:
    1) domain technical difficulty
    2) dominant bottleneck
    3) second-domain / interaction modifiers
    4) objective response modifiers

Engineering phase:
  - reads ONLY rrs_split == engineering
  - nested CV chooses among a small grid of clinically interpretable coefficients
  - outputs univariate diagnostics, domain/event-rate tables, nested OOF predictions,
    error audit, and frozen rule JSON

Shadow phase:
  - requires --rules from engineering
  - does NOT tune anything

Locked test:
  - intentionally unsupported
"""
import argparse, json, os, re, math, hashlib
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold

VERSION = "RRS-v3.3-BOTTLENECK"

SITE_KEYS = [
    "primary_pelvic_mass","pelvic_peritoneum","greater_omentum","lesser_omentum_lesser_sac",
    "small_bowel_mesenteric","colonic_mesenteric","paracolic_wall_peritoneum","diaphragmatic_peritoneum",
    "upper_abdominal_peritoneum","porta_hepatis","bowel_stomach_involvement",
    "pelvic_retroperitoneal_nodes","upper_abdominal_cardiophrenic_nodes","distant_metastasis"
]

DEFAULT_KW = {
    "mesenteric_root": ["肠系膜根", "系膜根部", "肠系膜上动脉", "sma root", "sma"],
    "diffuse_infiltrating": ["弥漫", "广泛", "融合", "大片", "饼状"],
    "deep_invasion": ["侵犯", "侵及", "浸润"],
    "mere_contact": ["密接", "分界不清", "界限不清", "受压", "邻近"],
    "small_bowel_organs": ["小肠", "回肠", "空肠", "十二指肠"],
    "stomach": ["胃", "胃壁", "胃大弯", "胃小弯"],
    "colon_rectum": ["直肠", "乙状结肠", "结肠", "直乙"],
    "porta": ["肝门", "肝十二指肠韧带", "胃肝韧带", "肝胃韧带"],
    "splenic_hilum": ["脾门"],
    "hepatorenal_recess": ["肝肾隐窝", "肝肾间隙"],
    "lesser_sac": ["网膜囊", "小网膜囊"],
    "gallbladder_fossa": ["胆囊窝", "肝裂", "叶间裂", "段间裂"],
    "major_vessel": ["腹腔干", "肝动脉", "胃左动脉", "门静脉", "肠系膜上动脉", "血管包绕", "包绕血管", "血管受累"],
    "diaphragm_deep": ["膈肌侵犯", "侵及膈肌", "膈肌内", "膈肌受累"],
    "bulky_node": ["淋巴结融合", "融合淋巴结", "淋巴结团块", "团块状淋巴结", "融合成团"],
    "obstruction": ["肠梗阻", "肠腔明显变窄", "肠腔狭窄", "肠管扩张", "胃出口梗阻"]
}

def norm_text(x):
    return str(x or "").lower().replace(" ", "")

def has_kw(text, kws):
    t = norm_text(text)
    return any(norm_text(k) in t for k in kws)

def safe_float(x):
    try:
        if x is None or (isinstance(x,float) and math.isnan(x)): return None
        return float(x)
    except Exception:
        return None

def clip(v,a,b): return max(a,min(b,v))

def load_jsonl(path):
    out={}
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            d=json.loads(line)
            out[str(d['patient_id'])]=d
    return out

def load_config(path):
    if path and os.path.exists(path):
        with open(path,'r',encoding='utf-8') as f: cfg=json.load(f)
        kw=cfg.get('keywords',DEFAULT_KW)
    else:
        cfg={"version":VERSION,"keywords":DEFAULT_KW}
        kw=DEFAULT_KW
    return cfg,kw

def detect_columns(df, label_col=None, split_col=None):
    cols=list(df.columns)
    lower={str(c).lower():c for c in cols}
    pid=None
    for c in ['patient_id','id','patientid','case_id']:
        if c in lower: pid=lower[c]; break
    if pid is None:
        for c in cols:
            if 'patient' in str(c).lower() and 'id' in str(c).lower(): pid=c; break
    if pid is None: raise ValueError(f"Cannot detect patient id column. Columns={cols}")

    if split_col is None:
        candidates=[]
        for c in cols:
            vals=set(str(v).strip().lower() for v in df[c].dropna().unique()[:50])
            if 'engineering' in vals or 'shadow_validation' in vals or 'locked_test' in vals:
                candidates.append(c)
        if candidates: split_col=candidates[0]
        else:
            for c in ['rrs_split','analysis_split','split']:
                if c in lower: split_col=lower[c]; break
    if split_col is None: raise ValueError(f"Cannot detect engineering/shadow split column. Columns={cols}")

    if label_col is None:
        for c in ['r0','r0_label','label','outcome','target','y']:
            if c in lower:
                label_col=lower[c]; break
    if label_col is None:
        # choose a column with R0/non-R0-like values
        for c in cols:
            vals=set(str(v).strip().lower() for v in df[c].dropna().unique()[:20])
            if any(v in vals for v in ['r0','non-r0','non_r0','残余≤1cm']):
                label_col=c; break
    if label_col is None: raise ValueError(f"Cannot detect label column. Use --label-col. Columns={cols}")
    return pid,label_col,split_col

def to_nonr0(v, label_col_name=''):
    # Return 1=non-R0, 0=R0
    if pd.isna(v): return np.nan
    s=str(v).strip().lower()
    if s in ['r0','1','1.0','true','yes'] and 'non' not in s:
        # If numeric label column is called r0, 1 means R0.
        if 'r0' in str(label_col_name).lower(): return 0
    if s in ['non-r0','non_r0','nonr0','残余≤1cm','残余<1cm','残余>1cm','0','0.0','false','no']:
        if s in ['0','0.0','false','no'] and 'r0' in str(label_col_name).lower(): return 1
        return 1
    if s=='r0': return 0
    # numeric fallback: if column is R0, 1=R0 0=non-R0; otherwise 1=non-R0
    try:
        x=float(v)
        if 'r0' in str(label_col_name).lower(): return int(1-x)
        return int(x)
    except: pass
    raise ValueError(f"Cannot interpret outcome value {v!r} in {label_col_name}")

def site_map(d):
    return {s['site']:s for s in d.get('sites',[])}

def post_present(s):
    return (s or {}).get('post_status')=='present'

def post_burden(s):
    x=(s or {}).get('post_burden_grade')
    try:return int(x) if x is not None else None
    except:return None

def post_rel(s):
    x=(s or {}).get('post_relation_grade')
    try:return int(x) if x is not None else None
    except:return None

def post_size(s): return safe_float((s or {}).get('post_max_size_mm'))
def pre_size(s): return safe_float((s or {}).get('pre_max_size_mm'))
def change(s): return (s or {}).get('radiologist_reported_change') or 'not_stated'
def evidence(s): return (s or {}).get('post_evidence') or ''

def objective_ratio(pre,post):
    pre=safe_float(pre); post=safe_float(post)
    if pre is None or post is None or pre<=0: return None
    return post/pre

def extract_components(d, kw):
    sm=site_map(d)
    # ---------- Mesentery domain ----------
    sb=sm.get('small_bowel_mesenteric',{})
    cm=sm.get('colonic_mesenteric',{})
    sb_ev=evidence(sb); cm_ev=evidence(cm)
    sb_b=post_burden(sb); cm_b=post_burden(cm)
    mes=0; mes_reason=[]
    if post_present(sb):
        mes=max(mes,1); mes_reason.append('small_bowel_mesenteric_present')
        if sb_b is not None:
            if sb_b>=2: mes=max(mes,2); mes_reason.append(f'small_bowel_burden{sb_b}')
            if sb_b>=3: mes=max(mes,3)
        if has_kw(sb_ev,kw['diffuse_infiltrating']):
            mes=max(mes,3); mes_reason.append('small_bowel_diffuse_or_confluent_wording')
        if has_kw(sb_ev,kw['mesenteric_root']):
            mes=4; mes_reason.append('mesenteric_root_or_SMA')
    if post_present(cm):
        # Colonic mesentery raw status is noisy; require burden >=2 or explicit diffuse/infiltrating wording
        if (cm_b is not None and cm_b>=2) or has_kw(cm_ev,kw['diffuse_infiltrating']):
            cmsev=2 if (cm_b or 0)==2 else 3
            mes=max(mes,cmsev); mes_reason.append(f'colonic_mesentery_confirmed_high_burden_{cm_b}')
        elif cm_b==1:
            mes=max(mes,1); mes_reason.append('colonic_mesentery_low_burden')
    if change(sb) in ('increased','new'): mes=clip(mes+1,0,4); mes_reason.append('mesentery_unfavorable_change')

    # ---------- Bowel/stomach domain ----------
    bw=sm.get('bowel_stomach_involvement',{})
    bw_ev=evidence(bw); rel=post_rel(bw) or 0
    bowel=0; bowel_reason=[]
    has_deep=has_kw(bw_ev,kw['deep_invasion'])
    has_small=has_kw(bw_ev,kw['small_bowel_organs'])
    has_stomach=has_kw(bw_ev,kw['stomach'])
    has_colon=has_kw(bw_ev,kw['colon_rectum'])
    has_obstruction=has_kw(bw_ev,kw['obstruction'])
    if rel<=1:
        bowel=0
        if rel==1: bowel_reason.append('contact_only_relation1')
    elif rel==2:
        # suspicious relation; only mild risk unless explicit invasion words
        bowel=1 if has_deep else 0
        bowel_reason.append('relation2_suspicious' if has_deep else 'relation2_without_explicit_invasion')
    elif rel>=3:
        if has_small or has_stomach:
            bowel=3
            bowel_reason.append('definite_small_bowel_or_stomach_involvement')
        elif has_colon:
            bowel=2
            bowel_reason.append('definite_colon_or_rectum_involvement')
        else:
            bowel=2
            bowel_reason.append('definite_bowel_relation3')
        if has_obstruction:
            bowel=4; bowel_reason.append('obstruction_or_marked_narrowing')
        if has_kw(bw_ev,kw['diffuse_infiltrating']) and has_deep:
            bowel=4; bowel_reason.append('diffuse_deep_bowel_pattern')
    # Status alone never exceeds 1 because QC showed raw status is noisy
    if bowel==0 and post_present(bw) and has_deep:
        bowel=1; bowel_reason.append('status_present_with_explicit_invasion_but_relation_not_high')

    # ---------- Upper abdomen domain ----------
    ua=sm.get('upper_abdominal_peritoneum',{})
    ph=sm.get('porta_hepatis',{})
    lo=sm.get('lesser_omentum_lesser_sac',{})
    ua_ev=evidence(ua); ph_ev=evidence(ph); lo_ev=evidence(lo)
    ua_b=post_burden(ua); lo_b=post_burden(lo); ph_b=post_burden(ph)
    upper=0; upper_reason=[]
    combined=' '.join([ua_ev,ph_ev,lo_ev])
    if has_kw(combined,kw['major_vessel']):
        upper=4; upper_reason.append('major_upper_abdominal_vessel_wording')
    if post_present(ph) or has_kw(combined,kw['porta']):
        upper=max(upper,3); upper_reason.append('porta_hepatis_or_gastrohepatic')
        if (post_size(ph) or 0)>=20 or (ph_b or 0)>=3: upper=max(upper,4); upper_reason.append('bulky_porta')
    if post_present(lo):
        if (lo_b or 0)>=2: upper=max(upper,2); upper_reason.append(f'lesser_sac_omentum_burden{lo_b}')
        elif lo_b==1: upper=max(upper,1); upper_reason.append('lesser_sac_omentum_low_burden')
    if post_present(ua):
        # Generic upper abdominal surface disease is low unless burden is high or high-risk subsite wording exists
        subrisk=0
        if has_kw(ua_ev,kw['splenic_hilum']) or has_kw(ua_ev,kw['hepatorenal_recess']) or has_kw(ua_ev,kw['gallbladder_fossa']): subrisk=2
        elif (ua_b or 0)>=3: subrisk=2
        elif (ua_b or 0)>=2: subrisk=1
        elif (ua_b or 0)==1: subrisk=0
        upper=max(upper,subrisk)
        if subrisk: upper_reason.append(f'upper_abd_surface_or_subsite_severity{subrisk}')
    if change(ph) in ('increased','new') or change(ua) in ('increased','new'):
        upper=clip(upper+1,0,4); upper_reason.append('upper_abd_unfavorable_change')

    # ---------- Diaphragm domain ----------
    dg=sm.get('diaphragmatic_peritoneum',{})
    dg_ev=evidence(dg); dg_b=post_burden(dg)
    diaphragm=0; diaphragm_reason=[]
    if post_present(dg):
        if has_kw(dg_ev,kw['diaphragm_deep']):
            diaphragm=3; diaphragm_reason.append('deep_diaphragm_muscle_involvement')
        elif (dg_b or 0)>=3 or has_kw(dg_ev,kw['diffuse_infiltrating']):
            diaphragm=2; diaphragm_reason.append('diffuse_or_confluent_diaphragm')
        elif (dg_b or 0)>=2:
            diaphragm=1; diaphragm_reason.append('moderate_diaphragm_burden')
        else:
            diaphragm=0; diaphragm_reason.append('focal_or_low_burden_surface_diaphragm')
    if change(dg) in ('increased','new'):
        diaphragm=clip(diaphragm+1,0,3); diaphragm_reason.append('diaphragm_unfavorable_change')

    # ---------- Nodes/distant domain ----------
    pn=sm.get('pelvic_retroperitoneal_nodes',{})
    un=sm.get('upper_abdominal_cardiophrenic_nodes',{})
    dm=sm.get('distant_metastasis',{})
    nodes=0; nodes_reason=[]
    for s,name,upperloc in [(pn,'pelvic_retroperitoneal',False),(un,'upper_cardiophrenic',True)]:
        if not post_present(s): continue
        sz=post_size(s); ev=evidence(s); ch=change(s)
        if sz is not None and sz<=10 and ch in ('stable','decreased','not_stated') and not has_kw(ev,kw['bulky_node']):
            sev=0
            nodes_reason.append(f'{name}_small_node_{sz:g}mm_zero')
        elif has_kw(ev,kw['bulky_node']) or (sz is not None and sz>20):
            sev=2
            nodes_reason.append(f'{name}_bulky_or_gt20mm')
        elif (sz is not None and sz>10) or ch in ('increased','new'):
            sev=1 if not upperloc else 2
            nodes_reason.append(f'{name}_suspicious_size_or_change')
        else:
            sev=0 if not upperloc else 1
            nodes_reason.append(f'{name}_present_unsized')
        nodes=max(nodes,sev)
    if post_present(dm):
        # Distant-metastasis raw extraction is weak; require explicit metastatic wording for full weight
        ev=evidence(dm)
        if '转移' in ev or 'metast' in ev.lower():
            nodes=max(nodes,3); nodes_reason.append('definite_distant_metastatic_wording')
        else:
            nodes=max(nodes,1); nodes_reason.append('distant_present_without_explicit_metastatic_wording')

    # ---------- Objective global response and burden ----------
    pm=sm.get('primary_pelvic_mass',{})
    pm_ratio=objective_ratio(pre_size(pm),post_size(pm))
    op=d.get('overall_peritoneal',{})
    per_ratio=objective_ratio(op.get('pre_group_max_size_mm'),op.get('post_group_max_size_mm'))
    post_per_size=safe_float(op.get('post_group_max_size_mm'))
    asc=d.get('abdominopelvic_fluid',{}).get('post',{}) or {}
    asc_grade=asc.get('grade')
    try: asc_grade=int(asc_grade) if asc_grade is not None else None
    except: asc_grade=None

    # Critical site counts; generic surface sites are not included
    domains=[mes,bowel,upper,diaphragm,nodes]
    sorted_dom=sorted(domains, reverse=True)
    dominant=sorted_dom[0]; second=sorted_dom[1]
    n_ge2=sum(v>=2 for v in domains)
    n_ge3=sum(v>=3 for v in domains)
    hard=int(any(v>=4 for v in domains))

    unfavorable=0; unfav_reason=[]
    # objective progression / several explicit unfavorable critical changes
    if pm_ratio is not None and pm_ratio>=1.2:
        unfavorable+=1; unfav_reason.append('primary_size_ratio>=1.2')
    if per_ratio is not None and per_ratio>=1.2:
        unfavorable+=1; unfav_reason.append('peritoneal_size_ratio>=1.2')
    critical_changes=[change(sb),change(cm),change(bw),change(ph),change(ua),change(dg),change(pn),change(un),change(dm)]
    if sum(c in ('increased','new') for c in critical_changes)>=2:
        unfavorable+=1; unfav_reason.append('>=2_critical_sites_increased_or_new')
    unfavorable=min(unfavorable,2)

    favorable=0; fav_reason=[]
    if pm_ratio is not None and pm_ratio<=0.5:
        favorable+=1; fav_reason.append('primary_size_ratio<=0.5')
    if per_ratio is not None and per_ratio<=0.5:
        favorable+=1; fav_reason.append('peritoneal_size_ratio<=0.5')
    # Only allow fluid as supportive, never alone
    if asc_grade==0 and favorable>0:
        favorable+=1; fav_reason.append('no_post_ascites_supports_response')
    favorable=min(favorable,2)

    global_burden=0; global_reason=[]
    if dominant>=2:
        if post_per_size is not None and post_per_size>=50:
            global_burden=1; global_reason.append('post_peritoneal_group_size>=50mm_with_critical_domain')
        # many post-present sites may indicate diffuse distribution, but only modify if a critical domain exists
        post_present_count=sum(1 for s in sm.values() if post_present(s))
        if post_present_count>=8:
            global_burden=max(global_burden,1); global_reason.append('>=8_post_present_sites_with_critical_domain')
    else:
        post_present_count=sum(1 for s in sm.values() if post_present(s))

    interaction=0; interaction_reason=[]
    if n_ge2>=2:
        interaction+=1; interaction_reason.append('>=2_domains_severity>=2')
    if mes>=2 and bowel>=2:
        interaction+=1; interaction_reason.append('mesentery_plus_bowel')
    if upper>=2 and diaphragm>=2:
        interaction+=1; interaction_reason.append('upper_abdomen_plus_diaphragm')
    interaction=min(interaction,2)

    return {
        'mesentery':mes,'bowel':bowel,'upper_abdomen':upper,'diaphragm':diaphragm,'nodes_distant':nodes,
        'dominant':dominant,'second':second,'n_domains_ge2':n_ge2,'n_domains_ge3':n_ge3,'hard':hard,
        'interaction':interaction,'unfavorable':unfavorable,'favorable':favorable,'global_burden':global_burden,
        'primary_size_ratio':pm_ratio,'peritoneal_size_ratio':per_ratio,'post_peritoneal_group_size_mm':post_per_size,
        'post_present_site_count':post_present_count,
        'reason_mes':';'.join(mes_reason),'reason_bowel':';'.join(bowel_reason),'reason_upper':';'.join(upper_reason),
        'reason_diaphragm':';'.join(diaphragm_reason),'reason_nodes':';'.join(nodes_reason),
        'reason_interaction':';'.join(interaction_reason),'reason_unfavorable':';'.join(unfav_reason),
        'reason_favorable':';'.join(fav_reason),'reason_global':';'.join(global_reason),
        'evidence_mes':(sb_ev+' '+cm_ev)[:800], 'evidence_bowel':bw_ev[:800],
        'evidence_upper':combined[:800], 'evidence_diaphragm':dg_ev[:800],
        'evidence_nodes':(evidence(pn)+' '+evidence(un)+' '+evidence(dm))[:800]
    }

def score_components(c, p):
    score=(p['dominant_weight']*c['dominant'] +
           p['second_weight']*c['second'] +
           p['multi_bonus']*c['interaction'] +
           p['hard_bonus']*c['hard'] +
           p['unfavorable_bonus']*c['unfavorable'] +
           p['global_bonus']*c['global_burden'])
    # Favorable response can only discount if there is no dominant severe barrier
    if c['dominant']<3:
        score -= p['favorable_credit']*c['favorable']
    return float(score)

def param_grid():
    # Restricted, interpretable coefficient grid. No per-site free weights.
    for vals in product(
        [2.0,2.5,3.0],    # dominant_weight
        [0.5,1.0,1.5],    # second_weight
        [0.5,1.0,1.5],    # multi_bonus
        [1.0,2.0,3.0],    # hard_bonus
        [0.5,1.0],        # unfavorable_bonus
        [0.5,1.0],        # favorable_credit
        [0.0,0.5,1.0],    # global_bonus
    ):
        yield dict(zip(['dominant_weight','second_weight','multi_bonus','hard_bonus','unfavorable_bonus','favorable_credit','global_bonus'],vals))

def auc(y,s):
    return roc_auc_score(y,s) if len(set(y))>1 else float('nan')

def best_params_inner(C, y, train_idx, seed=731):
    y=np.asarray(y)
    X=[C[i] for i in train_idx]
    yy=y[train_idx]
    skf=StratifiedKFold(n_splits=4,shuffle=True,random_state=seed)
    best=None; best_auc=-1
    grid=list(param_grid())
    for p in grid:
        fold=[]
        for a,b in skf.split(np.zeros(len(yy)),yy):
            scores=[score_components(X[i],p) for i in b]
            fold.append(auc(yy[b],scores))
        m=float(np.nanmean(fold))
        if m>best_auc+1e-12:
            best_auc=m; best=p
    return best,best_auc

def choose_threshold(y,s):
    fpr,tpr,thr=roc_curve(y,s)
    j=tpr-fpr
    i=int(np.nanargmax(j))
    return float(thr[i])

def summarize_cm(y,pred):
    y=np.asarray(y); pred=np.asarray(pred)
    tn=int(((y==0)&(pred==0)).sum()); fp=int(((y==0)&(pred==1)).sum())
    fn=int(((y==1)&(pred==0)).sum()); tp=int(((y==1)&(pred==1)).sum())
    return {'tn':tn,'fp':fp,'fn':fn,'tp':tp,
            'sensitivity':tp/(tp+fn) if tp+fn else None,
            'specificity':tn/(tn+fp) if tn+fp else None,
            'ppv':tp/(tp+fp) if tp+fp else None,
            'npv':tn/(tn+fn) if tn+fn else None,
            'accuracy':(tp+tn)/len(y) if len(y) else None}

def univariate_table(df):
    rows=[]
    fields=['mesentery','bowel','upper_abdomen','diaphragm','nodes_distant','dominant','second','n_domains_ge2','n_domains_ge3','hard','interaction','unfavorable','favorable','global_burden','post_present_site_count']
    for f in fields:
        try:a=auc(df['y_non_r0'].values,df[f].values)
        except:a=np.nan
        rows.append({'feature':f,'auroc':a,'mean_R0':df.loc[df.y_non_r0==0,f].mean(),'mean_nonR0':df.loc[df.y_non_r0==1,f].mean()})
    for f in ['primary_size_ratio','peritoneal_size_ratio','post_peritoneal_group_size_mm']:
        sub=df[['y_non_r0',f]].dropna()
        try:a=auc(sub.y_non_r0.values,sub[f].values) if len(sub)>5 else np.nan
        except:a=np.nan
        rows.append({'feature':f,'auroc':a,'mean_R0':sub.loc[sub.y_non_r0==0,f].mean(),'mean_nonR0':sub.loc[sub.y_non_r0==1,f].mean(),'n_nonmissing':len(sub)})
    return pd.DataFrame(rows).sort_values('auroc',ascending=False,na_position='last')

def event_rate_tables(df,outdir):
    allrows=[]
    for f in ['mesentery','bowel','upper_abdomen','diaphragm','nodes_distant','dominant','n_domains_ge2','interaction','hard']:
        for val,g in df.groupby(f,dropna=False):
            allrows.append({'feature':f,'value':val,'n':len(g),'non_r0_n':int(g.y_non_r0.sum()),'non_r0_rate':float(g.y_non_r0.mean())})
    pd.DataFrame(allrows).to_csv(Path(outdir)/'engineering_event_rates_by_domain.csv',index=False,encoding='utf-8-sig')

    # Prespecified clinically plausible interactions only
    pairs=[('mesentery','bowel'),('mesentery','upper_abdomen'),('mesentery','diaphragm'),('upper_abdomen','diaphragm'),('bowel','nodes_distant')]
    rows=[]
    for a,b in pairs:
        A=(df[a]>=2).astype(int); B=(df[b]>=2).astype(int)
        for av in [0,1]:
            for bv in [0,1]:
                g=df[(A==av)&(B==bv)]
                if len(g): rows.append({'pair':f'{a}+{b}','a_ge2':av,'b_ge2':bv,'n':len(g),'non_r0_n':int(g.y_non_r0.sum()),'non_r0_rate':float(g.y_non_r0.mean())})
    pd.DataFrame(rows).to_csv(Path(outdir)/'engineering_prespecified_interactions.csv',index=False,encoding='utf-8-sig')

def engineering(df, components, outdir, config_path):
    y=df['y_non_r0'].to_numpy(dtype=int)
    C=components
    outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=20261015)
    oof=np.zeros(len(df),float); oof_thr=np.zeros(len(df),float); fold_id=np.zeros(len(df),int)
    fold_params=[]
    for fold,(tr,va) in enumerate(outer.split(np.zeros(len(y)),y),1):
        p,inner_auc=best_params_inner(C,y,tr,seed=900+fold)
        train_scores=np.array([score_components(C[i],p) for i in tr])
        thr=choose_threshold(y[tr],train_scores)
        val_scores=np.array([score_components(C[i],p) for i in va])
        oof[va]=val_scores; oof_thr[va]=thr; fold_id[va]=fold
        fold_params.append({'fold':fold,'inner_cv_auc':inner_auc,'threshold_from_outer_train':thr,**p})
    oof_auc=auc(y,oof); oof_ap=average_precision_score(y,oof)
    oof_pred=(oof>=oof_thr).astype(int)
    oof_cm=summarize_cm(y,oof_pred)

    # final params selected using 5-fold CV on ALL engineering, still without shadow
    grid=list(param_grid())
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=4242)
    best=None;bestcv=-1
    for p in grid:
        fas=[]
        for tr,va in skf.split(np.zeros(len(y)),y):
            scores=[score_components(C[i],p) for i in va]
            fas.append(auc(y[va],scores))
        m=float(np.nanmean(fas))
        if m>bestcv+1e-12: bestcv=m;best=p
    # Create OOF scores under final fixed params for threshold selection
    fixed_oof=np.zeros(len(df),float)
    for tr,va in skf.split(np.zeros(len(y)),y):
        fixed_oof[va]=[score_components(C[i],best) for i in va]
    frozen_thr=choose_threshold(y,fixed_oof)
    fixed_auc=auc(y,fixed_oof); fixed_ap=average_precision_score(y,fixed_oof)

    out=df.copy()
    out['nested_oof_score']=oof; out['nested_oof_fold']=fold_id; out['nested_oof_threshold']=oof_thr
    out['nested_oof_pred_non_r0']=oof_pred
    out['error_type']=np.where((y==0)&(oof_pred==0),'TN',np.where((y==0)&(oof_pred==1),'FP',np.where((y==1)&(oof_pred==0),'FN','TP')))
    out['fixed_formula_oof_score']=fixed_oof
    out.to_csv(Path(outdir)/'engineering_nested_oof_predictions_and_error_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(fold_params).to_csv(Path(outdir)/'engineering_outer_fold_selected_params.csv',index=False,encoding='utf-8-sig')
    univariate_table(df).to_csv(Path(outdir)/'engineering_univariate_signal.csv',index=False,encoding='utf-8-sig')
    event_rate_tables(df,outdir)

    rules={
        'version':VERSION,
        'formula_family':'dominant-bottleneck + second-domain + prespecified interaction + objective response',
        'selected_params':best,
        'frozen_threshold':frozen_thr,
        'selection_cv_auc_mean':bestcv,
        'fixed_formula_5fold_oof_auc':fixed_auc,
        'fixed_formula_5fold_oof_auprc':fixed_ap,
        'config_sha256': hashlib.sha256(open(config_path,'rb').read()).hexdigest() if config_path and os.path.exists(config_path) else None,
        'locked_test_evaluated':False,'shadow_evaluated':False
    }
    with open(Path(outdir)/'RRS_v3_3_RULES_FROZEN.json','w',encoding='utf-8') as f: json.dump(rules,f,ensure_ascii=False,indent=2)
    summary={
        'version':VERSION,'engineering_n':len(df),'r0':int((y==0).sum()),'non_r0':int((y==1).sum()),
        'nested_5fold_oof_auroc_non_r0':oof_auc,'nested_5fold_oof_auprc_non_r0':oof_ap,
        'nested_oof_confusion_using_outer_train_thresholds':oof_cm,
        'final_fixed_formula_5fold_oof_auroc_non_r0':fixed_auc,'final_fixed_formula_5fold_oof_auprc_non_r0':fixed_ap,
        'frozen_threshold_from_fixed_formula_oof':frozen_thr,
        'selected_params':best,
        'target_auc_0_80_reached':bool(oof_auc>=0.80),
        'shadow_evaluated':False,'locked_test_evaluated':False,
        'interpretation_note':'If nested OOF AUROC remains well below 0.80, the limiting factor is likely information granularity/reliability of the existing 14-site representation or outcome-label noise, not simple weight tuning.'
    }
    with open(Path(outdir)/'engineering_summary.json','w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def shadow(df,components,outdir,rules_path):
    if not rules_path: raise ValueError('--rules is required for shadow phase')
    rules=json.load(open(rules_path,encoding='utf-8'))
    p=rules['selected_params']; thr=float(rules['frozen_threshold'])
    y=df['y_non_r0'].to_numpy(dtype=int)
    s=np.array([score_components(c,p) for c in components])
    pred=(s>=thr).astype(int)
    summary={'version':VERSION,'shadow_n':len(df),'r0':int((y==0).sum()),'non_r0':int((y==1).sum()),
             'shadow_auroc_non_r0':auc(y,s),'shadow_auprc_non_r0':average_precision_score(y,s),
             'threshold':thr,'confusion':summarize_cm(y,pred),'locked_test_evaluated':False}
    out=df.copy();out['score']=s;out['pred_non_r0']=pred
    out.to_csv(Path(outdir)/'shadow_predictions.csv',index=False,encoding='utf-8-sig')
    with open(Path(outdir)/'shadow_summary.json','w',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
    flag=Path(outdir)/'SHADOW_ALREADY_EVALUATED.flag'; flag.write_text('shadow evaluated once\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['engineering','shadow'],required=True)
    ap.add_argument('--jsonl',required=True)
    ap.add_argument('--labels',required=True)
    ap.add_argument('--config',default='RRS_v3_3_CONFIG.json')
    ap.add_argument('--rules',default=None)
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--label-col',default=None)
    ap.add_argument('--split-col',default=None)
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)
    cases=load_jsonl(args.jsonl); cfg,kw=load_config(args.config)
    lab=pd.read_csv(args.labels,dtype=str)
    pidcol,labelcol,splitcol=detect_columns(lab,args.label_col,args.split_col)
    lab[pidcol]=lab[pidcol].astype(str).str.strip()
    lab['_split']=lab[splitcol].astype(str).str.strip().str.lower()
    lab['y_non_r0']=lab[labelcol].apply(lambda x:to_nonr0(x,labelcol))
    target='engineering' if args.phase=='engineering' else 'shadow_validation'
    sub=lab[lab['_split']==target].copy()
    if len(sub)==0: raise ValueError(f'No rows with {splitcol}={target}. Observed={lab[splitcol].value_counts(dropna=False).to_dict()}')
    rows=[]; comps=[]
    for _,r in sub.iterrows():
        pid=str(r[pidcol])
        if pid not in cases: raise KeyError(f'patient_id {pid} not in JSONL')
        c=extract_components(cases[pid],kw); comps.append(c)
        row={'patient_id':pid,'y_non_r0':int(r['y_non_r0'])}; row.update(c); rows.append(row)
    df=pd.DataFrame(rows).reset_index(drop=True)
    # Write transparent component table before any outcome-based optimization
    df.to_csv(Path(args.outdir)/f'{args.phase}_components_before_formula.csv',index=False,encoding='utf-8-sig')
    if args.phase=='engineering': engineering(df,comps,args.outdir,args.config)
    else:
        flag=Path(args.outdir)/'SHADOW_ALREADY_EVALUATED.flag'
        if flag.exists(): raise RuntimeError(f'Shadow has already been evaluated in {args.outdir}. Do not iterate on shadow.')
        shadow(df,comps,args.outdir,args.rules)

if __name__=='__main__': main()
