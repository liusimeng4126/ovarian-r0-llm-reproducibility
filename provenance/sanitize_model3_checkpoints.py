from pathlib import Path
import hashlib, json, tarfile, torch

import argparse
parser = argparse.ArgumentParser(description='Sanitize locally supplied trusted Model 3 checkpoints.')
parser.add_argument('--source-archive', type=Path, required=True)
parser.add_argument('--release-root', type=Path, required=True)
arguments = parser.parse_args()
root = arguments.release_root
src = arguments.source_archive
seeds={0:420042,1:420043,2:420044}
manifest=[]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
with tarfile.open(src) as tar:
    for idx,seed in seeds.items():
        member=f'MODEL3_DEV392_FINAL_RUN/FROZEN_MODEL3_AFTER_DEV392/seed_{idx}/model.pt'
        raw=tar.extractfile(member).read()
        import io
        obj=torch.load(io.BytesIO(raw),map_location='cpu')
        clean={k:v for k,v in obj.items() if k not in {'fit_patient_ids','development392_features_sha256','locked_internal_test_used','external_used'}}
        out=root/'model3/checkpoints_release'/('model3_seed%d_release.pt'%seed)
        out.parent.mkdir(parents=True,exist_ok=True)
        with open(out,'wb') as fh: torch.save(clean,fh)
        # compare every tensor value against source
        for k,v in obj['state_dict'].items():
            assert torch.equal(v,clean['state_dict'][k]), k
        manifest.append({'seed':seed,'source_member':member,'source_sha256':hashlib.sha256(raw).hexdigest(),'release_path':str(out.relative_to(root)),'release_sha256':sha(out),'removed_keys':['fit_patient_ids','development392_features_sha256','locked_internal_test_used','external_used']})
(root/'provenance/MODEL3_CHECKPOINT_RELEASE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(root/'provenance/sanitize_model3_checkpoints.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
