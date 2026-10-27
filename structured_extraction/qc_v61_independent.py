#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent, outcome-blinded QC for frozen V6.1 extraction.

Purpose
-------
1) Randomly sample cases ONLY from model_dev_372_new (never prompt-dev 1-20,
   never final_test_100).
2) Create a clinician review workbook containing the full PRE/POST reports,
   extracted site-level fields, and blank manual-standard columns.
3) After manual review, optionally score field-level agreement.

No surgical-result label is read or written by this script.
"""
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation

SITE_ORDER = [
    "primary_pelvic_mass","pelvic_peritoneum","greater_omentum","lesser_omentum_lesser_sac",
    "small_bowel_mesenteric","colonic_mesenteric","paracolic_wall_peritoneum",
    "diaphragmatic_peritoneum","upper_abdominal_peritoneum","porta_hepatis",
    "bowel_stomach_involvement","pelvic_retroperitoneal_nodes",
    "upper_abdominal_cardiophrenic_nodes","distant_metastasis",
]
STATUS_ENUM = ["present","absent","uncertain","not_mentioned","not_evaluable"]
CHANGE_ENUM = ["decreased","increased","stable","mixed","resolved","new","not_evaluable","not_stated"]
MANUAL_FIELDS = ["pre_status","post_status","pre_burden_grade","post_burden_grade",
                 "pre_max_size_mm","post_max_size_mm","pre_relation_grade","post_relation_grade",
                 "radiologist_reported_change"]

def read_jsonl(path: Path):
    out=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip(): out.append(json.loads(line))
    return out

def parse_pairs(path: Path):
    text=path.read_text(encoding='utf-8-sig')
    pat=re.compile(r'(?m)^\s*[【\[]?\s*(\d+)\s*([abAB])\s*[】\]]?\s*$')
    ms=list(pat.finditer(text)); pairs={}
    for i,m in enumerate(ms):
        pid=str(int(m.group(1))); arm=m.group(2).lower(); end=ms[i+1].start() if i+1<len(ms) else len(text)
        pairs.setdefault(pid,{})[arm]=text[m.end():end].strip()
    return pairs

def eligible_ids(split_xlsx: Path):
    wb=load_workbook(split_xlsx,read_only=True,data_only=True)
    ws=wb['model_dev_372_new']
    return [str(int(r[0])) for r in ws.iter_rows(min_row=2,values_only=True) if r[0] is not None]

def site_map(p): return {x['site']:x for x in p.get('sites',[])}

def style_header(ws):
    fill=PatternFill('solid',fgColor='1F4E78')
    for c in ws[1]:
        c.font=Font(bold=True,color='FFFFFF'); c.fill=fill; c.alignment=Alignment(horizontal='center',vertical='center')
    ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions

def add_dropdowns(ws, header_index, nrows):
    status_dv=DataValidation(type='list', formula1='"'+','.join(STATUS_ENUM)+'"', allow_blank=True)
    change_dv=DataValidation(type='list', formula1='"'+','.join(CHANGE_ENUM)+'"', allow_blank=True)
    burden_dv=DataValidation(type='list', formula1='"0,1,2,3"', allow_blank=True)
    relation_dv=DataValidation(type='list', formula1='"0,1,2,3"', allow_blank=True)
    for dv in (status_dv,change_dv,burden_dv,relation_dv): ws.add_data_validation(dv)
    for h in ['manual_pre_status','manual_post_status']:
        col=ws.cell(1,header_index[h]+1).column_letter; status_dv.add(f'{col}2:{col}{nrows}')
    col=ws.cell(1,header_index['manual_radiologist_reported_change']+1).column_letter; change_dv.add(f'{col}2:{col}{nrows}')
    for h in ['manual_pre_burden_grade','manual_post_burden_grade']:
        col=ws.cell(1,header_index[h]+1).column_letter; burden_dv.add(f'{col}2:{col}{nrows}')
    for h in ['manual_pre_relation_grade','manual_post_relation_grade']:
        col=ws.cell(1,header_index[h]+1).column_letter; relation_dv.add(f'{col}2:{col}{nrows}')

def make_review(jsonl:Path, reports:Path, split_xlsx:Path, out:Path, n:int, seed:int, ids_file=None):
    rows=read_jsonl(jsonl); byid={str(x['patient_id']):x for x in rows}; pairs=parse_pairs(reports)
    frame=eligible_ids(split_xlsx)
    if ids_file:
        requested=[x.strip() for x in Path(ids_file).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
        bad=[x for x in requested if x not in frame]
        if bad: raise ValueError(f'Locked QC IDs outside model_dev_372_new: {bad}')
        missing=[x for x in requested if x not in byid or set(pairs.get(x,{}))!={'a','b'}]
        if missing: raise ValueError(f'Locked QC cases missing extraction/report pair: {missing}')
        sample=sorted(requested,key=int); n=len(sample)
        sample_note=f'locked ID file={ids_file}'
    else:
        eligible=[x for x in frame if x in byid and set(pairs.get(x,{}))=={'a','b'}]
        if len(eligible)<n: raise ValueError(f'Only {len(eligible)} eligible extracted report pairs found; need {n}.')
        rng=random.Random(seed); sample=sorted(rng.sample(eligible,n), key=int)
        sample_note=f'random seed={seed}'
    wb=Workbook(); ws=wb.active; ws.title='site_review'
    headers=['patient_id','site','pre_status','post_status','pre_burden_grade','post_burden_grade',
             'pre_max_size_mm','post_max_size_mm','pre_relation_grade','post_relation_grade',
             'radiologist_reported_change','pre_evidence','post_evidence'] + \
            [f'manual_{x}' for x in MANUAL_FIELDS] + ['error_type','reviewer_notes']
    ws.append(headers)
    for pid in sample:
        p=byid[pid]; sm=site_map(p)
        for site in SITE_ORDER:
            s=sm.get(site,{})
            ws.append([pid,site,s.get('pre_status'),s.get('post_status'),s.get('pre_burden_grade'),s.get('post_burden_grade'),
                       s.get('pre_max_size_mm'),s.get('post_max_size_mm'),s.get('pre_relation_grade'),s.get('post_relation_grade'),
                       s.get('radiologist_reported_change'),s.get('pre_evidence',''),s.get('post_evidence','')] + ['']*(len(MANUAL_FIELDS)+2))
    style_header(ws); hi={c.value:i for i,c in enumerate(ws[1])}; add_dropdowns(ws,hi,ws.max_row)
    for col,w in {'A':11,'B':34,'C':16,'D':16,'K':22,'L':55,'M':55}.items(): ws.column_dimensions[col].width=w
    for row in ws.iter_rows(min_row=2):
        row[11].alignment=Alignment(wrap_text=True,vertical='top'); row[12].alignment=Alignment(wrap_text=True,vertical='top')

    cr=wb.create_sheet('case_reports')
    cr.append(['patient_id','PRE_report','POST_report'])
    for pid in sample: cr.append([pid,pairs[pid]['a'],pairs[pid]['b']])
    style_header(cr); cr.column_dimensions['A'].width=11; cr.column_dimensions['B'].width=90; cr.column_dimensions['C'].width=90
    for row in cr.iter_rows(min_row=2):
        row[1].alignment=Alignment(wrap_text=True,vertical='top'); row[2].alignment=Alignment(wrap_text=True,vertical='top')

    cs=wb.create_sheet('case_summary')
    cs.append(['patient_id','pre_report_incomplete','post_report_incomplete','n_hotfix_actions','manual_case_has_any_error','case_notes'])
    for pid in sample:
        p=byid[pid]; dq=p.get('data_quality',{}); hf=p.get('_v6_1_hotfix',{})
        cs.append([pid,bool(dq.get('pre_report_incomplete')),bool(dq.get('post_report_incomplete')),hf.get('n_actions',0),'',''])
    style_header(cs)

    ins=wb.create_sheet('instructions')
    instructions=[
      ['Independent QC instructions'],
      ['1. This workbook deliberately contains NO surgical-result label. Review only source report versus extraction.'],
      ['2. Read full PRE/POST in case_reports first, then review all 14 rows for that patient in site_review.'],
      ['3. Fill manual_* fields with your gold-standard interpretation. Fill not_mentioned explicitly when appropriate; do not leave it blank if reviewed.'],
      ['4. Numeric max_size_mm is in mm. The scoring script accepts a configurable tolerance (default ±1 mm).'],
      ['5. Suggested error_type: missed_site / false_positive / benign_as_malignant / relation_error / size_error / change_error / other.'],
      ['6. QC is an accuracy audit, not a new prompt-development set. Isolated errors are recorded, not used to tune the frozen prompt.'],
      [f'QC sample: n={n}, {sample_note}, frame=model_dev_372_new only (excludes 1-20 and final test 100).'],
    ]
    for x in instructions: ins.append(x)
    ins.column_dimensions['A'].width=130

    out.parent.mkdir(parents=True,exist_ok=True); wb.save(out)
    (out.parent/'qc_sample_ids.txt').write_text('\n'.join(sample)+'\n',encoding='utf-8')
    print(f'QC workbook: {out}'); print(f'Sampled IDs: {out.parent/"qc_sample_ids.txt"}')

def norm(v):
    if v is None: return None
    if isinstance(v,str):
        v=v.strip()
        if not v: return None
        if v.lower() in {'none','null','na','n/a'}: return None
    return v

def numeric_equal(a,b,tol):
    a=norm(a); b=norm(b)
    if a is None or b is None: return a is None and b is None
    try: return abs(float(a)-float(b))<=tol
    except Exception: return str(a)==str(b)

def score_review(path:Path, out:Path, tol:float):
    wb=load_workbook(path,data_only=True); ws=wb['site_review']
    headers=[c.value for c in ws[1]]; idx={h:i for i,h in enumerate(headers)}
    stats={f:{'reviewed':0,'correct':0} for f in MANUAL_FIELDS}; site_stats={}; case_err={}
    for row in ws.iter_rows(min_row=2,values_only=True):
        pid=str(row[idx['patient_id']]); site=row[idx['site']]; all_correct=True; any_reviewed=False
        for f in MANUAL_FIELDS:
            mv=norm(row[idx[f'manual_{f}']])
            if mv is None: continue
            any_reviewed=True; ev=norm(row[idx[f]])
            ok=numeric_equal(ev,mv,tol) if f.endswith('_mm') else str(ev)==str(mv)
            stats[f]['reviewed']+=1; stats[f]['correct']+=int(ok)
            site_stats.setdefault(site,{'reviewed':0,'correct':0}); site_stats[site]['reviewed']+=1; site_stats[site]['correct']+=int(ok)
            all_correct &= ok
        if any_reviewed and not all_correct: case_err[pid]=1
    payload={'size_tolerance_mm':tol,
             'field_metrics':[{'field':f,**s,'accuracy':(s['correct']/s['reviewed'] if s['reviewed'] else None)} for f,s in stats.items()],
             'site_metrics':{k:{**v,'accuracy':(v['correct']/v['reviewed'] if v['reviewed'] else None)} for k,v in site_stats.items()},
             'n_cases_with_any_scored_error':len(case_err)}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'QC score: {out}')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--jsonl',help='all_patients_v6_1.jsonl; required when making QC workbook')
    ap.add_argument('--reports',help='Full or 372-case report TXT; required when making QC workbook')
    ap.add_argument('--split-xlsx',help='cohort_split_492_locked.xlsx; required when making QC workbook')
    ap.add_argument('--out',default='qc_v61/qc_review_40.xlsx')
    ap.add_argument('--n',type=int,default=40); ap.add_argument('--seed',type=int,default=20261012)
    ap.add_argument('--ids',help='Optional locked QC ID txt; recommended for the formal audit')
    ap.add_argument('--score',help='Clinician-filled qc_review_40.xlsx')
    ap.add_argument('--score-out',default='qc_v61/qc_metrics.json'); ap.add_argument('--size-tolerance-mm',type=float,default=1.0)
    a=ap.parse_args()
    if a.score: score_review(Path(a.score),Path(a.score_out),a.size_tolerance_mm)
    else:
        if not a.jsonl or not a.reports or not a.split_xlsx: ap.error('--jsonl, --reports and --split-xlsx are required unless --score is used')
        make_review(Path(a.jsonl),Path(a.reports),Path(a.split_xlsx),Path(a.out),a.n,a.seed,a.ids)
if __name__=='__main__': main()
