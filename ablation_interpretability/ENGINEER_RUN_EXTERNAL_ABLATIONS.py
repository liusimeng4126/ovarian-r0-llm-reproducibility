from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from model3_test_common import *
from model3_core.data import NumericPreprocessor
from model3_core.development import apply_variant
from model3_core.model import Model3, ModelConfig

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--dev-run',required=True); ap.add_argument('--blind-input-dir',default='MODEL3_BLIND_INPUTS_READY'); ap.add_argument('--ablation-dir',default='MODEL3_TEST100_EXTERNAL80_RETURN/03_EXTERNAL80_ABLATIONS'); ap.add_argument('--outdir',default='MODEL3_TEST100_EXTERNAL80_RETURN/03_EXTERNAL80_ABLATION_PREDICTIONS'); ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); args=ap.parse_args()
 import torch
 dev=Path(args.dev_run).resolve(); blind=Path(args.blind_input_dir).resolve(); abd=Path(args.ablation_dir).resolve(); out=Path(args.outdir).resolve(); out.mkdir(parents=True,exist_ok=True)
 schema=read_json(dev/'schema.json'); raw=load_raw_npz(blind/'inputs'/'EXTERNAL80_FEATURES_NO_LABELS.npz'); batch=raw_arrays_to_feature_batch(raw,schema)
 pre=NumericPreprocessor.from_dict(read_json(dev/'FROZEN_MODEL3_AFTER_DEV392'/'preprocessor.json')); base=pre.transform(batch); device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
 for variant in ('noSiteID','noDelta','PostOnly','noAttention'):
  transformed=apply_variant(base,variant); x={k:torch.as_tensor(v,dtype=torch.long if 'categorical' in k else torch.float32,device=device) for k,v in transformed.items()}; ps=[]
  for si in range(3):
   ck=abd/variant/f'seed_{si}'/'model.pt'; payload=torch.load(ck,map_location=device,weights_only=True); model=Model3(ModelConfig.from_dict(payload['model_config'])).to(device); model.load_state_dict(payload['state_dict']); model.eval()
   with torch.inference_mode(): logits,_=model(x); ps.append(torch.sigmoid(logits).cpu().numpy())
  ps=np.stack(ps,axis=1); p=ps.mean(axis=1); rows=[{'patient_id':int(pid),'p_r0':float(p[i]),'seed0_p_r0':float(ps[i,0]),'seed1_p_r0':float(ps[i,1]),'seed2_p_r0':float(ps[i,2])} for i,pid in enumerate(batch.patient_id)]
  pred=out/f'MODEL3_EXTERNAL80_{variant}_BLIND_PREDICTIONS.csv'; write_csv(pred,list(rows[0]),rows); write_json(out/f'MODEL3_EXTERNAL80_{variant}_BLIND_MANIFEST.json',{'variant':variant,'n':80,'outcomes_used':False,'prediction_sha256':sha256(pred),'primary_model_replacement_allowed':False})
  print('PASS',variant,sha256(pred))
 print('='*80); print('PASS - EXTERNAL80 ABLATION BLIND PREDICTIONS COMPLETE'); print('='*80)
if __name__=='__main__': main()
