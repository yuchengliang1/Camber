import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from runtime import resolve_device
from pathlib import Path
import argparse, json, sys, hashlib, importlib.util
import numpy as np, torch
CODE = Path(str(_PACKAGE / 'code/00_shared/models'))
sys.path.insert(0, str(CODE))
import run_nested_hierarchical_pair_anchor_v25 as ex
import hierarchical_pair_anchor_models_v25 as base_model
FULL = ex._symbols(base_model)['FULL_VARIANT']

def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}

def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--subjects', required=True)
    ap.add_argument('--base-seed', type=int, default=27029001)
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    mm = base_model
    cfgs = json.loads(a.spec.read_text())['configs']
    subjects = [int(x) for x in a.subjects.split(',')]
    a.out_root.mkdir(parents=True, exist_ok=False)
    write(a.out_root / 'manifest.json', {'subjects': subjects, 'configs': cfgs})
    dev = resolve_device(a.device)
    for s in subjects:
        g = f'sub{s - 29100:02d}'
        d = load(a.data_root / g / 'paired_mi.npz')
        for cfg in cfgs:
            name = cfg['name']
            root = a.out_root / g / name
            root.mkdir(parents=True, exist_ok=False)
            prob = np.full((150, 3), np.nan, np.float32)
            pred = np.full(150, -1, np.int64)
            rows = []
            prov = {}
            for (f, b) in enumerate(sorted(np.unique(d['block']).astype(int).tolist())):
                tr = np.flatnonzero(d['block'] != b).astype(np.int64)
                te = np.flatnonzero(d['block'] == b).astype(np.int64)
                path = root / FULL / f'fold_{f}' / 'outer'
                seed = a.base_seed + s * 100000 + f * 1000 + 999
                ex._fit_once(path, variant=FULL, config=cfg, train_indices=tr, evaluation_indices=te, seed=seed, eeg_raw=d['eeg'], meg_raw=d['meg'], y=d['y'], device=dev, provenance=prov, model_mod=mm)
                with np.load(path / 'predictions.npz', allow_pickle=False) as z:
                    ii = np.asarray(z['indices'], np.int64)
                    pp = np.asarray(z['probability'], np.float32)
                    pr = np.asarray(z['prediction'], np.int64)
                prob[ii] = pp
                pred[ii] = pr
                acc = float(100 * np.mean(pr == d['y'][ii]))
                rows.append({'fold': f, 'held_out_block': b, 'accuracy_percent': acc})
                print(json.dumps({'subject': s, 'config': name, 'fold': f, 'accuracy_percent': acc}), flush=True)
            metrics = ex.metrics(d['y'], prob)
            write(root / 'summary.json', {'subject': s, 'config_name': name, 'config': cfg, 'metrics': metrics, 'folds': rows})
            print(json.dumps({'subject': s, 'config': name, 'summary_accuracy_percent': metrics['accuracy_percent']}), flush=True)
    print(json.dumps({'status': 'PASS', 'subjects': subjects}), flush=True)
if __name__ == '__main__':
    main()
