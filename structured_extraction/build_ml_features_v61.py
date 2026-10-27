#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build compact machine-learning tables from frozen V6.1 structured extraction.

Design
------
- Feature generation never reads the surgical endpoint.
- Endpoint/split are exported in a separate labels_and_split.csv.
- Model 2 uses features_v61_wide.csv (one patient per row).
- Model 3 uses site_features_v61_long.csv + global_features_v61.csv.
- The wide and long tables contain the SAME V6.1 information, only reshaped.
- Categorical columns remain strings. Encode them INSIDE each training fold later;
  do not one-hot the entire dataset before cross-validation.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from typing import Any, Dict
from openpyxl import load_workbook

SITE_ORDER=[
"primary_pelvic_mass","pelvic_peritoneum","greater_omentum","lesser_omentum_lesser_sac",
"small_bowel_mesenteric","colonic_mesenteric","paracolic_wall_peritoneum","diaphragmatic_peritoneum",
"upper_abdominal_peritoneum","porta_hepatis","bowel_stomach_involvement","pelvic_retroperitoneal_nodes",
"upper_abdominal_cardiophrenic_nodes","distant_metastasis"]

STATUS_VALUES={"present","absent","uncertain","not_mentioned","not_evaluable"}
CHANGE_VALUES={"decreased","increased","stable","mixed","resolved","new","not_evaluable","not_stated"}


def read_jsonl(p: str):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]

def finite_num(x):
    if x is None or x=='': return None
    try:
        v=float(x); return v if math.isfinite(v) else None
    except Exception: return None

def transition(pre,post):
    if pre=='present' and post=='present': return 'persistent_present'
    if pre=='present' and post=='absent': return 'resolved'
    if pre=='absent' and post=='present': return 'new'
    if pre=='absent' and post=='absent': return 'persistent_absent'
    if 'not_evaluable' in {pre,post}: return 'not_evaluable'
    if 'not_mentioned' in {pre,post}: return 'has_not_mentioned'
    return 'uncertain_transition'

def delta(a,b):
    a=finite_num(a); b=finite_num(b)
    return None if a is None or b is None else b-a

def rel_delta(a,b):
    a=finite_num(a); b=finite_num(b)
    return None if a in (None,0) or b is None else (b-a)/a

def site_features(s: Dict[str,Any]):
    pre=s.get('pre_status','not_mentioned'); post=s.get('post_status','not_mentioned')
    ch=s.get('radiologist_reported_change','not_stated')
    return {
        'pre_status': pre,
        'post_status': post,
        'status_transition': transition(pre,post),
        'pre_burden_grade': finite_num(s.get('pre_burden_grade')),
        'post_burden_grade': finite_num(s.get('post_burden_grade')),
        'burden_delta': delta(s.get('pre_burden_grade'),s.get('post_burden_grade')),
        'pre_max_size_mm': finite_num(s.get('pre_max_size_mm')),
        'post_max_size_mm': finite_num(s.get('post_max_size_mm')),
        'max_size_delta_mm': delta(s.get('pre_max_size_mm'),s.get('post_max_size_mm')),
        'max_size_relative_change': rel_delta(s.get('pre_max_size_mm'),s.get('post_max_size_mm')),
        'pre_relation_grade': finite_num(s.get('pre_relation_grade')),
        'post_relation_grade': finite_num(s.get('post_relation_grade')),
        'relation_delta': delta(s.get('pre_relation_grade'),s.get('post_relation_grade')),
        'radiologist_reported_change': ch,
    }

def fluid_features(prefix,block):
    block=block or {}; pre=block.get('pre') or {}; post=block.get('post') or {}
    return {
        f'{prefix}_pre_status':pre.get('status','not_mentioned'),
        f'{prefix}_post_status':post.get('status','not_mentioned'),
        f'{prefix}_status_transition':transition(pre.get('status','not_mentioned'),post.get('status','not_mentioned')),
        f'{prefix}_pre_grade':finite_num(pre.get('grade')),
        f'{prefix}_post_grade':finite_num(post.get('grade')),
        f'{prefix}_grade_delta':delta(pre.get('grade'),post.get('grade')),
    }

def global_features(p):
    op=p.get('overall_peritoneal') or {}
    sites=p.get('sites') or []
    d={
        'pre_group_max_size_mm':finite_num(op.get('pre_group_max_size_mm')),
        'post_group_max_size_mm':finite_num(op.get('post_group_max_size_mm')),
        'group_max_size_delta_mm':delta(op.get('pre_group_max_size_mm'),op.get('post_group_max_size_mm')),
        'group_max_size_relative_change':rel_delta(op.get('pre_group_max_size_mm'),op.get('post_group_max_size_mm')),
        'pre_report_incomplete':int(bool((p.get('data_quality') or {}).get('pre_report_incomplete'))),
        'post_report_incomplete':int(bool((p.get('data_quality') or {}).get('post_report_incomplete'))),
    }
    d.update(fluid_features('abdominopelvic_fluid',p.get('abdominopelvic_fluid')))
    d.update(fluid_features('pleural_effusion',p.get('pleural_effusion')))
    for phase in ('pre','post'):
        d[f'{phase}_present_site_count']=sum(s.get(f'{phase}_status')=='present' for s in sites)
        d[f'{phase}_uncertain_site_count']=sum(s.get(f'{phase}_status')=='uncertain' for s in sites)
        d[f'{phase}_absent_site_count']=sum(s.get(f'{phase}_status')=='absent' for s in sites)
        d[f'{phase}_not_mentioned_site_count']=sum(s.get(f'{phase}_status')=='not_mentioned' for s in sites)
        d[f'{phase}_not_evaluable_site_count']=sum(s.get(f'{phase}_status')=='not_evaluable' for s in sites)
    trs=[transition(s.get('pre_status'),s.get('post_status')) for s in sites]
    d['persistent_present_site_count']=sum(x=='persistent_present' for x in trs)
    d['resolved_site_count']=sum(x=='resolved' for x in trs)
    d['new_site_count']=sum(x=='new' for x in trs)
    d['persistent_absent_site_count']=sum(x=='persistent_absent' for x in trs)
    for ch in ['decreased','increased','stable','mixed','resolved','new']:
        d[f'{ch}_site_count']=sum(s.get('radiologist_reported_change')==ch for s in sites)
    return d

def write_csv(path,rows,fieldnames=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('',encoding='utf-8'); return
    if fieldnames is None:
        keys=[]; seen=set()
        for r in rows:
            for k in r:
                if k not in seen: seen.add(k); keys.append(k)
        fieldnames=keys
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def load_labels_split(xlsx):
    wb=load_workbook(xlsx,read_only=True,data_only=True); ws=wb['all_492']
    hdr=[x.value for x in ws[1]]; ix={h:i for i,h in enumerate(hdr)}; rows=[]
    for r in ws.iter_rows(min_row=2,values_only=True):
        if r[ix['ID']] is None: continue
        outcome=str(r[ix['减瘤结果']]).strip(); role=str(r[ix['cohort_role']]).strip()
        rows.append({'patient_id':str(int(r[ix['ID']])), 'actual_outcome':outcome,
                     'R0_binary':1 if outcome=='R0' else 0,
                     'cohort_role':role,
                     'model_role':'final_test' if role=='final_test_locked' else 'development'})
    return rows

def feature_dictionary(wide_headers,long_headers,global_headers):
    rows=[]
    categorical_suffixes=('pre_status','post_status','status_transition','radiologist_reported_change')
    for h in wide_headers:
        if h=='patient_id': continue
        kind='categorical' if h.endswith(categorical_suffixes) else 'numeric'
        group='global' if h.startswith('global__') else h.split('__',1)[0]
        rows.append({'table':'features_v61_wide.csv','feature':h,'type':kind,'group':group,
                     'description':'Derived only from frozen V6.1 structured extraction.'})
    # add long/global schema once so computer collaborators know the site-model input.
    for h in long_headers:
        if h in {'patient_id','site'}: continue
        kind='categorical' if h in {'pre_status','post_status','status_transition','radiologist_reported_change'} else 'numeric'
        rows.append({'table':'site_features_v61_long.csv','feature':h,'type':kind,'group':'per_site',
                     'description':'Same site-level information used in the wide table, reshaped for site-aware modeling.'})
    for h in global_headers:
        if h=='patient_id': continue
        kind='categorical' if h.endswith(('status','transition')) else 'numeric'
        rows.append({'table':'global_features_v61.csv','feature':h,'type':kind,'group':'global',
                     'description':'Patient-level summary derived from the same frozen V6.1 record.'})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--jsonl',required=True); ap.add_argument('--split-xlsx',required=True); ap.add_argument('--outdir',default='ml_v61')
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    patients=read_jsonl(a.jsonl); wide=[]; long=[]; globals_rows=[]
    for p in patients:
        pid=str(p['patient_id']); sm={s['site']:s for s in p.get('sites',[])}; wr={'patient_id':pid}
        for site in SITE_ORDER:
            sf=site_features(sm.get(site,{'site':site}))
            long.append({'patient_id':pid,'site':site,**sf})
            for k,v in sf.items(): wr[f'{site}__{k}']=v
        gf=global_features(p); globals_rows.append({'patient_id':pid,**gf})
        for k,v in gf.items(): wr[f'global__{k}']=v
        wide.append(wr)
    labels=load_labels_split(a.split_xlsx)
    write_csv(out/'features_v61_wide.csv',wide); write_csv(out/'site_features_v61_long.csv',long); write_csv(out/'global_features_v61.csv',globals_rows); write_csv(out/'labels_and_split.csv',labels)
    wide_headers=list(wide[0].keys()) if wide else []; long_headers=list(long[0].keys()) if long else []; global_headers=list(globals_rows[0].keys()) if globals_rows else []
    write_csv(out/'feature_dictionary.csv',feature_dictionary(wide_headers,long_headers,global_headers))
    pids={str(x['patient_id']) for x in patients}; lpids={x['patient_id'] for x in labels}
    manifest={'n_extracted':len(patients),'n_wide_rows':len(wide),'n_wide_columns':len(wide_headers),
              'n_site_long_rows':len(long),'n_site_long_columns':len(long_headers),'expected_site_long_rows':len(patients)*len(SITE_ORDER),
              'n_global_columns':len(global_headers),'missing_label_ids':sorted(pids-lpids,key=int),
              'missing_extraction_ids':sorted(lpids-pids,key=int),'site_order':SITE_ORDER,
              'endpoint_used_to_generate_features':False,
              'categorical_encoding_note':'Keep categorical strings as-is in this table. Fit categorical encoding only within development-set CV folds.',
              'model2_input':'features_v61_wide.csv','model3_input':'site_features_v61_long.csv + global_features_v61.csv',
              'fair_comparison_note':'Model 2 and Model 3 receive the same frozen V6.1 information; only the data shape/model architecture differs.'}
    (out/'feature_build_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
