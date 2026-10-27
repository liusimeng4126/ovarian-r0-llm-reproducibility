#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse,concurrent.futures,csv,hashlib,json,os,re,threading,time
from datetime import datetime,timezone
from pathlib import Path
from typing import Any,Dict
from model1_v2_prompt import SYSTEM_PROMPT,USER_TEMPLATE,PROMPT_SHA256
try:
    from openai import OpenAI
except Exception:
    OpenAI=None
VERSION='MODEL1-V2-CLINICALLY-INSTRUCTED-QWEN38MAX-v2.0-2026-10-19'
DEFAULT_MODEL='qwen3.8-max'
DEFAULT_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
TEMPERATURE=0
ENABLE_THINKING=False

def now(): return datetime.now(timezone.utc).isoformat()
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def parse_pairs(text:str):
    ms=list(re.finditer(r'(?m)^\s*[【\[]?\s*(\d+)\s*([abAB])\s*[】\]]?\s*$',text))
    if not ms: raise ValueError('No patient headers like 1a / 1b found')
    out={}
    for i,m in enumerate(ms):
        start=m.end(); end=ms[i+1].start() if i+1<len(ms) else len(text)
        pid=str(int(m.group(1))); arm=m.group(2).lower(); out.setdefault(pid,{})[arm]=text[start:end].strip()
    return out

def parse_json(text:str)->Dict[str,Any]:
    t=(text or '').strip(); t=re.sub(r'^```(?:json)?\s*','',t,flags=re.I); t=re.sub(r'\s*```$','',t)
    try: obj=json.loads(t)
    except Exception:
        a=t.find('{'); b=t.rfind('}')
        if a<0 or b<=a: raise
        obj=json.loads(t[a:b+1])
    if not isinstance(obj,dict) or 'p_r0' not in obj: raise ValueError('JSON must be object with p_r0')
    p=float(obj['p_r0'])
    if not (0<=p<=1): raise ValueError(f'p_r0 out of range: {p}')
    return {'p_r0':p}

def call_one(client,model,pre,post,retries):
    user=USER_TEMPLATE.format(pre=pre,post=post); last=None
    for k in range(retries):
        try:
            resp=client.chat.completions.create(
                model=model,temperature=TEMPERATURE,
                messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':user}],
                extra_body={'enable_thinking':False},
            )
            raw=resp.choices[0].message.content or ''; obj=parse_json(raw); usage=getattr(resp,'usage',None)
            u={x:int(getattr(usage,x,0) or 0) if usage else 0 for x in ['prompt_tokens','completion_tokens','total_tokens']}
            return obj,raw,u
        except Exception as e:
            last=e
            if k+1<retries: time.sleep(min(2**k,8))
    raise RuntimeError(f'API/JSON failed after {retries} attempts: {last}')

def load_jsonl(path):
    if not path.exists(): return []
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--outdir',required=True)
    ap.add_argument('--model',default=DEFAULT_MODEL); ap.add_argument('--base-url',default=os.getenv('DASHSCOPE_BASE_URL',DEFAULT_BASE_URL)); ap.add_argument('--concurrency',type=int,default=3); ap.add_argument('--max-retries',type=int,default=4); ap.add_argument('--resume',action='store_true'); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--expected-n',type=int); a=ap.parse_args()
    inp=Path(a.input).resolve(); out=Path(a.outdir).resolve(); out.mkdir(parents=True,exist_ok=True)
    rawbytes=inp.read_bytes(); pairs=parse_pairs(rawbytes.decode('utf-8-sig')); ids=sorted(pairs,key=lambda x:int(x)); miss=[x for x in ids if set(pairs[x])!={'a','b'}]
    if miss: raise ValueError(f'Missing PRE/POST pairs: {miss[:20]}')
    if a.expected_n is not None and len(ids)!=a.expected_n: raise ValueError(f'Expected {a.expected_n} patients, found {len(ids)}')
    print(f'VERSION={VERSION}'); print(f'PROMPT_SHA256={PROMPT_SHA256}'); print(f'INPUT_SHA256={sha_bytes(rawbytes)}'); print(f'Validated {len(ids)} paired patients. First={ids[0]} Last={ids[-1]}')
    if a.dry_run: print('DRY RUN PASS: no API call made.'); return
    if OpenAI is None: raise RuntimeError('openai package missing; run 01_INSTALL_DEPENDENCIES.cmd')
    key=os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not key: raise RuntimeError('No API key. In CMD: set DASHSCOPE_API_KEY=YOUR_KEY')
    predp=out/'model1_v2_predictions.jsonl'; tracep=out/'model1_v2_trace.jsonl'; csvp=out/'model1_v2_predictions.csv'; manp=out/'model1_v2_manifest.json'
    if predp.exists() and not a.resume: raise RuntimeError(f'{predp} exists. Do not overwrite frozen predictions; use --resume only if the previous run was interrupted.')
    existing=load_jsonl(predp) if a.resume else []; done={str(x['patient_id']) for x in existing}; pending=[x for x in ids if x not in done]; lock=threading.Lock(); errors={}; total={k:0 for k in ['prompt_tokens','completion_tokens','total_tokens']}; started=now()
    for r in existing:
        for k in total: total[k]+=int((r.get('usage') or {}).get(k,0) or 0)
    def persist():
        m={'version':VERSION,'model':a.model,'base_url':a.base_url,'temperature':TEMPERATURE,'enable_thinking':ENABLE_THINKING,'prompt_sha256':PROMPT_SHA256,'input_file':str(inp),'input_sha256':sha_bytes(rawbytes),'n_requested':len(ids),'n_completed':len(done),'usage':total,'errors':errors,'started_utc':started,'updated_utc':now()}
        manp.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    persist()
    client=OpenAI(api_key=key,base_url=a.base_url)
    def work(pid):
        try:
            obj,raw,u=call_one(client,a.model,pairs[pid]['a'],pairs[pid]['b'],a.max_retries)
            rec={'patient_id':pid,'p_r0':obj['p_r0'],'usage':u}
            with lock:
                with predp.open('a',encoding='utf-8') as f: f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                with tracep.open('a',encoding='utf-8') as f: f.write(json.dumps({'patient_id':pid,'raw_response':raw},ensure_ascii=False)+'\n')
                done.add(pid)
                for k in total: total[k]+=u[k]
                persist(); print(f'[{len(done)}/{len(ids)}] patient {pid} p_r0={obj["p_r0"]:.4f}',flush=True)
        except Exception as e:
            with lock: errors[pid]=repr(e); persist(); print(f'ERROR patient {pid}: {e}',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,a.concurrency)) as ex:
        futs=[ex.submit(work,pid) for pid in pending]
        for f in concurrent.futures.as_completed(futs): f.result()
    recs=sorted(load_jsonl(predp),key=lambda x:int(x['patient_id']))
    if len(recs)!=len(ids): raise RuntimeError(f'Only {len(recs)}/{len(ids)} completed. Resolve errors and rerun with --resume.')
    with csvp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['patient_id','p_r0','prompt_tokens','completion_tokens','total_tokens']); w.writeheader()
        for r in recs:
            u=r.get('usage') or {}; w.writerow({'patient_id':r['patient_id'],'p_r0':r['p_r0'],'prompt_tokens':u.get('prompt_tokens',0),'completion_tokens':u.get('completion_tokens',0),'total_tokens':u.get('total_tokens',0)})
    persist(); print('DONE',csvp); print('USAGE',json.dumps(total,ensure_ascii=False))
if __name__=='__main__': main()
