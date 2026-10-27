from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
from model3_test_common import *
from model3_core.data import NumericPreprocessor
from model3_core.model import Model3, ModelConfig
from model3_core.development import occlude_site

EXPECTED={
'features392':'a449f8ae9c8d241ab62022524365a7ad62c321105f7e84d074c5fcb84ee2a996',
'schema':'5e5c1960f9cc444878b9d52739a592e064ade8366594fd6276156932bdad26a1',
'preprocessor':'2985788c65ece4283f00557a032f9b9239d5518f142627a3c7a0a7a8ec353d2d',
'checkpoint0':'309d03e18b907109abdd97944e262b7bcf1f5cd4bdd3091d33e7689a902f9a2c',
'checkpoint1':'13d78609a6c6d6f7f2288bf758a4d18e363f35c8bea1e45fe303bba805ec563e',
'checkpoint2':'b5f354c73e0a2dbdf263901ade0e4536d216f177d622fd585555e7b35e7c5c93'}

def tensorize(x,device):
 import torch
 return {k:torch.as_tensor(v,dtype=torch.long if 'categorical' in k else torch.float32,device=device) for k,v in x.items()}

def infer_cohort(raw_path,dev_run,outdir,cohort,device_name='auto'):
 import torch
 dev=Path(dev_run); out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
 schema=read_json(dev/'schema.json'); raw=load_raw_npz(raw_path); batch=raw_arrays_to_feature_batch(raw,schema)
 pre=NumericPreprocessor.from_dict(read_json(dev/'FROZEN_MODEL3_AFTER_DEV392'/'preprocessor.json')); transformed=pre.transform(batch)
 device=torch.device('cuda' if device_name=='cuda' or (device_name=='auto' and torch.cuda.is_available()) else 'cpu')
 x=tensorize(transformed,device); probs=[]; atts=[]; models=[]
 for si in range(3):
   ck=dev/'FROZEN_MODEL3_AFTER_DEV392'/f'seed_{si}'/'model.pt'; payload=torch.load(ck,map_location=device,weights_only=True)
   model=Model3(ModelConfig.from_dict(payload['model_config'])).to(device); model.load_state_dict(payload['state_dict']); model.eval(); models.append(model)
   with torch.inference_mode(): logits,att=model(x); probs.append(torch.sigmoid(logits).cpu().numpy()); atts.append(att.cpu().numpy())
 pseed=np.stack(probs,axis=1); p=pseed.mean(axis=1); att=np.stack(atts).mean(axis=0)
 threshold=float(read_json(dev/'FROZEN_MODEL3_AFTER_DEV392'/'FREEZE_MANIFEST.json')['threshold']['threshold'])
 rows=[]
 for i,pid in enumerate(batch.patient_id): rows.append({'patient_id':int(pid),'p_r0':float(p[i]),'pred_r0_frozen_threshold':int(p[i]>=threshold),'seed0_p_r0':float(pseed[i,0]),'seed1_p_r0':float(pseed[i,1]),'seed2_p_r0':float(pseed[i,2])})
 pred=out/f'MODEL3_{cohort}_FULL_BLIND_PREDICTIONS.csv'; write_csv(pred,list(rows[0]),rows)
 attrows=[]
 for i,pid in enumerate(batch.patient_id): attrows.append({'patient_id':int(pid),**{schema['site_order'][j]:float(att[i,j]) for j in range(14)}})
 write_csv(out/f'MODEL3_{cohort}_FULL_ATTENTION.csv',['patient_id']+schema['site_order'],attrows)
 occ=[]
 for j,site in enumerate(schema['site_order']):
   xo=occlude_site(x,j); op=[]
   for model in models:
    with torch.inference_mode(): logits,_=model(xo); op.append(torch.sigmoid(logits).cpu().numpy())
   po=np.stack(op).mean(axis=0); occ.append(p-po)
 occ=np.stack(occ,axis=1); orows=[]
 for i,pid in enumerate(batch.patient_id): orows.append({'patient_id':int(pid),**{schema['site_order'][j]:float(occ[i,j]) for j in range(14)}})
 write_csv(out/f'MODEL3_{cohort}_FULL_SITE_OCCLUSION_DELTA_P.csv',['patient_id']+schema['site_order'],orows)
 man={'cohort':cohort,'n':len(rows),'threshold':threshold,'blind_prediction_sha256':sha256(pred),'outcomes_used':False,'checkpoint_sha256':[sha256(dev/'FROZEN_MODEL3_AFTER_DEV392'/f'seed_{i}'/'model.pt') for i in range(3)]}
 write_json(out/f'MODEL3_{cohort}_FULL_BLIND_MANIFEST.json',man); return man

def validate_dev(dev):
 dev=Path(dev)
 checks={'features392':sha256(dev/'features392.npz'),'schema':sha256(dev/'schema.json'),'preprocessor':sha256(dev/'FROZEN_MODEL3_AFTER_DEV392'/'preprocessor.json')}
 for i in range(3): checks[f'checkpoint{i}']=sha256(dev/'FROZEN_MODEL3_AFTER_DEV392'/f'seed_{i}'/'model.pt')
 bad={k:(checks[k],EXPECTED[k]) for k in EXPECTED if checks[k]!=EXPECTED[k]}
 if bad: raise ValueError(f'Frozen Development392 run hash mismatch: {bad}')
 freeze=read_json(dev/'FROZEN_MODEL3_AFTER_DEV392'/'FREEZE_MANIFEST.json')
 if freeze.get('locked_internal_test_used') or freeze.get('external_used'): raise ValueError('Freeze manifest says test/external was previously used')
 return checks

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--dev-run',required=True); ap.add_argument('--blind-input-dir',default='MODEL3_BLIND_INPUTS_READY'); ap.add_argument('--outdir',default='MODEL3_TEST100_EXTERNAL80_RETURN'); ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); args=ap.parse_args()
 dev=Path(args.dev_run).resolve(); blind=Path(args.blind_input_dir).resolve(); out=Path(args.outdir).resolve(); out.mkdir(parents=True,exist_ok=True)
 checks=validate_dev(dev)
 i=load_raw_npz(blind/'inputs'/'INTERNAL100_FEATURES_NO_LABELS.npz'); e=load_raw_npz(blind/'inputs'/'EXTERNAL80_FEATURES_NO_LABELS.npz')
 if len(i['patient_id'])!=100 or len(set(i['patient_id'].tolist()))!=100: raise ValueError('Internal100 blind input invalid')
 if len(e['patient_id'])!=80 or len(set(e['patient_id'].tolist()))!=80: raise ValueError('External80 blind input invalid')
 m1=infer_cohort(blind/'inputs'/'INTERNAL100_FEATURES_NO_LABELS.npz',dev,out/'01_INTERNAL100_FULL','INTERNAL100',args.device)
 m2=infer_cohort(blind/'inputs'/'EXTERNAL80_FEATURES_NO_LABELS.npz',dev,out/'02_EXTERNAL80_FULL','EXTERNAL80',args.device)
 write_json(out/'00_PREFLIGHT_PRIMARY.json',{'status':'PASS','dev_hashes':checks,'internal100_n':100,'external80_n':80,'outcomes_used':False})
 print('='*80); print('PASS - PRIMARY BLIND INFERENCE COMPLETE'); print(json.dumps({'internal100':m1,'external80':m2},indent=2)); print('='*80)
if __name__=='__main__': main()
