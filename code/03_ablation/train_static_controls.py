import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from runtime import resolve_device
from pathlib import Path
from types import SimpleNamespace
import importlib.util, json, sys
import numpy as np
import torch
sys.path.insert(0, str(_PACKAGE / 'code/00_shared/models'))
import run_nested_hierarchical_pair_anchor_v25 as ex
import hierarchical_pair_anchor_models_v25 as base
S = Path(str(_PACKAGE / 'code/00_shared/static_control_models.py'))
sp = importlib.util.spec_from_file_location('static_control_models', S)
static = importlib.util.module_from_spec(sp)
sp.loader.exec_module(static)
DATA = Path(str(_PACKAGE / 'data'))
SOURCE = Path(str(_RESULTS / '01_training_validation/full_and_ablations'))
OUT = Path(str(_RESULTS / '03_ablation/learned_static'))
FULL = base.FULL_VARIANT
CONTROLS = static.PRIMARY_CONTROLS

def read(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}

def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, sort_keys=True))

def make_module():
    return static

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    mf = json.loads((SOURCE / 'experiment_manifest.json').read_text())
    mod = make_module()
    ex._symbols(mod)
    cfg = mf['config']
    device = resolve_device()
    OUT.mkdir(exist_ok=False)
    write(OUT / 'experiment_manifest.json', {'data_root': str(DATA), 'parent_training_root': str(SOURCE), 'config': cfg, 'variants': list(CONTROLS), 'reference': FULL, 'outer_folds': 5, 'inner_search': False, 'capacity': static.capacity_record()})
    all_subject = []
    for sid in range(29101, 29113):
        group = f'sub{sid - 29100:02d}'
        d = read(DATA / group / 'paired_mi.npz')
        splits = json.loads((_PACKAGE / 'data' / group / 'split.json').read_text())['folds']
        g = OUT / group
        g.mkdir()
        probs = {name: np.full((150, 3), np.nan, np.float32) for name in CONTROLS}
        provenance = {}
        for (fold, split) in enumerate(splits):
            tr = np.asarray(split['train_indices'], np.int64)
            te = np.asarray(split['test_indices'], np.int64)
            ref = SOURCE / group / FULL / f'fold_{fold}' / 'outer'
            spec = json.loads((ref / 'specification.json').read_text())
            assert tr.tolist() == spec['train_indices'] and te.tolist() == spec['evaluation_indices']
            seed = int(spec['seed'])
            if sid == 29101 and fold == 0:
                path = g / static.REFERENCE_FULL / f'fold_{fold}' / 'outer'
                ex._fit_once(path, variant=static.REFERENCE_FULL, config=cfg, train_indices=tr, evaluation_indices=te, seed=seed, eeg_raw=d['eeg'], meg_raw=d['meg'], y=d['y'], device=device, provenance=provenance, model_mod=mod)
                new = read(path / 'predictions.npz')['probability']
                old = read(ref / 'predictions.npz')['probability']
                maximum = float(np.max(np.abs(new - old)))
                write(OUT / 'reference_equivalence.json', {'subject': sid, 'fold': fold, 'same_seed': True, 'same_split': True, 'max_probability_abs_difference': maximum, 'reference_prediction_equal': bool(np.array_equal(new.argmax(1), old.argmax(1)))})
                if maximum > 1e-06:
                    raise RuntimeError('Static reference differs from Full; do not reuse reference')
                print(json.dumps({'event': 'reference_equivalence', 'max_abs_difference': maximum}), flush=True)
            for name in CONTROLS:
                path = g / name / f'fold_{fold}' / 'outer'
                result = ex._fit_once(path, variant=name, config=cfg, train_indices=tr, evaluation_indices=te, seed=seed, eeg_raw=d['eeg'], meg_raw=d['meg'], y=d['y'], device=device, provenance=provenance, model_mod=mod)
                z = read(path / 'predictions.npz')
                assert np.array_equal(z['indices'], te)
                probs[name][te] = z['probability']
                print(json.dumps({'subject': sid, 'fold': fold, 'variant': name, 'accuracy_percent': result['outcome']['accuracy_percent']}), flush=True)
        ref_oof = read(SOURCE / group / FULL / 'oof_predictions.npz')
        assert np.array_equal(d['y'], ref_oof['y']) and np.isfinite(ref_oof['probability']).all()
        summary = {'subject': sid, 'reference_full_accuracy_percent': float(100 * np.mean(ref_oof['probability'].argmax(1) == d['y'])), 'controls': {}}
        for (name, p) in probs.items():
            assert np.isfinite(p).all()
            path = g / name / 'oof_predictions.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, probability=p, prediction=p.argmax(1), y=d['y'], block=d['block'], covered=np.ones(150, bool))
            summary['controls'][name] = {'accuracy_percent': float(100 * np.mean(p.argmax(1) == d['y']))}
        write(g / 'summary.json', summary)
        all_subject.append(summary)
        print(json.dumps({'subject': sid, 'status': 'complete', 'control_accuracy_percent': {k: v['accuracy_percent'] for (k, v) in summary['controls'].items()}}), flush=True)
    aggregate = {'status': 'PASS', 'subjects': 12, 'reference_full_mean_percent': float(np.mean([r['reference_full_accuracy_percent'] for r in all_subject])), 'controls_mean_percent': {name: float(np.mean([r['controls'][name]['accuracy_percent'] for r in all_subject])) for name in CONTROLS}}
    write(OUT / 'summary.json', aggregate)
    print(json.dumps(aggregate), flush=True)
if __name__ == '__main__':
    main()
