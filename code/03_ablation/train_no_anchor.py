import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from runtime import resolve_device
from pathlib import Path
from types import SimpleNamespace, MethodType
import json, hashlib, sys
import numpy as np
import torch
sys.path.insert(0, str(_PACKAGE))
import train_models as original
ex = original.ex
base = original.base_model
FULL = base.FULL_VARIANT
DATA = Path(str(_PACKAGE / 'data'))
SOURCE = Path(str(_RESULTS / '01_training_validation/full_and_ablations'))
OUT = Path(str(_RESULTS / '03_ablation/no_anchor'))

def no_anchor_details(self, eeg, meg):
    detail = base.HierarchicalPairAnchorModelV25.forward_with_details(self, eeg, meg)
    p = detail['structured_probability']
    detail['log_probability'] = p.clamp_min(base.EPS).log()
    detail['mix_weight'] = torch.ones_like(detail['mix_weight'])
    return detail

def model_module(model_base):
    symbols = ex._symbols(model_base)

    def make_model(variant=FULL, dropout=0.0):
        if variant != FULL:
            raise ValueError('No-Anchor experiment accepts Full architecture only')
        model = model_base.make_model(variant, dropout)
        model.requires_anchor_pretraining = False
        model.requires_anchor_mixing_calibration = True
        model.anchor.requires_grad_(False)
        model.anchor_mix_residual.requires_grad_(False)
        model._anchor_frozen = True
        model.forward_with_details = MethodType(no_anchor_details, model)
        return model
    symbols['make_model'] = make_model
    symbols['MODEL_FAMILY'] = str(symbols['MODEL_FAMILY']) + '_from_scratch_no_anchor'
    return SimpleNamespace(**symbols)

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = json.loads((SOURCE / 'experiment_manifest.json').read_text())
    cfg = manifest['config']
    model_mod = model_module(base)
    ex._symbols(model_mod)
    OUT.mkdir(parents=True, exist_ok=False)
    original.write(OUT / 'experiment_manifest.json', {'data_root': str(DATA), 'parent_training_root': str(SOURCE), 'variant': 'full_structured_without_anchor', 'config': cfg, 'outer_folds': 5, 'anchor_pretraining': False})
    device = resolve_device()
    aggregate = []
    for sid in range(29101, 29113):
        group = f'sub{sid - 29100:02d}'
        data_path = DATA / group / 'paired_mi.npz'
        d = original.load_data(data_path)
        split = json.loads((_PACKAGE / 'data' / group / 'split.json').read_text())
        root = OUT / group
        root.mkdir()
        provenance = {}
        prob = np.full((150, 3), np.nan, np.float32)
        covered = np.zeros(150, bool)
        fold_results = []
        for (f, fold) in enumerate(split['folds']):
            tr = np.asarray(fold['train_indices'], np.int64)
            te = np.asarray(fold['test_indices'], np.int64)
            main_spec = json.loads((SOURCE / group / FULL / f'fold_{f}' / 'outer' / 'specification.json').read_text())
            assert tr.tolist() == main_spec['train_indices'] and te.tolist() == main_spec['evaluation_indices']
            seed = int(main_spec['seed'])
            fit_dir = root / f'fold_{f}' / 'outer'
            result = ex._fit_once(fit_dir, variant=FULL, config=cfg, train_indices=tr, evaluation_indices=te, seed=seed, eeg_raw=d['eeg'], meg_raw=d['meg'], y=d['y'], device=device, provenance=provenance, model_mod=model_mod)
            with np.load(fit_dir / 'predictions.npz', allow_pickle=False) as z:
                ix = z['indices']
                assert np.array_equal(ix, te) and (not covered[ix].any())
                prob[ix] = z['probability']
                covered[ix] = True
                assert np.allclose(z['probability'], z['structured_probability'], atol=1e-06)
                assert np.allclose(z['mix_weight'], 1, atol=1e-07)
            pretrain = json.loads((fit_dir / 'anchor_pretraining.json').read_text())
            assert pretrain['applicable'] is False
            fold_results.append({'fold': f, 'accuracy_percent': result['outcome']['accuracy_percent'], 'fit_path': str(fit_dir)})
            print(json.dumps({'subject': sid, 'fold': f, 'accuracy_percent': result['outcome']['accuracy_percent']}), flush=True)
        assert covered.all() and np.isfinite(prob).all()
        p = root / 'oof_predictions.npz'
        np.savez_compressed(p, probability=prob, prediction=prob.argmax(1), y=d['y'], block=d['block'], covered=covered)
        row = {'subject': sid, 'accuracy_percent': float(100 * np.mean(prob.argmax(1) == d['y'])), 'folds': fold_results, 'oof_path': str(p)}
        original.write(root / 'summary.json', row)
        aggregate.append(row)
        print(json.dumps({'subject': sid, 'status': 'complete', 'accuracy_percent': row['accuracy_percent']}), flush=True)
    original.write(OUT / 'summary.json', {'status': 'PASS', 'subject_results': aggregate, 'mean_subject_accuracy_percent': float(np.mean([r['accuracy_percent'] for r in aggregate]))})
    print(json.dumps({'status': 'PASS', 'mean_accuracy_percent': float(np.mean([r['accuracy_percent'] for r in aggregate]))}), flush=True)
if __name__ == '__main__':
    main()
