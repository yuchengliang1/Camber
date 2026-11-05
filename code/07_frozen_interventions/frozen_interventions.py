import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from runtime import resolve_device
from pathlib import Path
import importlib.util, json, sys
import numpy as np
import torch
ROOT = Path(str(_PACKAGE))
CODE = Path(str(_PACKAGE / 'code/00_shared/models'))
INTERVENTION_CODE = Path(str(_PACKAGE / 'code/07_frozen_interventions'))
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(INTERVENTION_CODE))
import hierarchical_pair_anchor_models_v25 as base
import run_interventions as intervention
DATA = _PACKAGE / 'data'
RUN = _RESULTS / '01_training_validation/full_and_ablations'
OUT = _RESULTS / '07_frozen_interventions'
manifest = json.loads((RUN / 'experiment_manifest.json').read_text())
mm = base
config = manifest['config']
device = resolve_device()
REPEATS = 100
CONDITIONS = intervention.CONDITIONS

def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}

def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False)
    write(OUT / 'manifest.json', {'data_root': str(DATA), 'model_root': str(RUN), 'repeats': REPEATS, 'conditions': list(CONDITIONS), 'training': False, 'waveform_mutation': False, 'task_codes_modified': False, 'reference': 'CAMBER frozen feature intervention implementation'})
    rows = []
    for sid in range(29101, 29113):
        g = f'sub{sid - 29100:02d}'
        d = npz(DATA / g / 'paired_mi.npz')
        all_prob = np.empty((150, REPEATS, len(CONDITIONS), 3), np.float32)
        audit = []
        coverage = np.zeros(150, bool)
        for fold in range(5):
            test = np.flatnonzero(d['block'] == fold)
            fit = RUN / g / base.FULL_VARIANT / f'fold_{fold}' / 'outer'
            scales = npz(fit / 'scalers.npz')
            model = mm.make_model(base.FULL_VARIANT, dropout=config['dropout'])
            model.load_state_dict(torch.load(fit / 'model.pt', map_location='cpu', weights_only=True), strict=True)
            model.to(device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            eeg = torch.as_tensor(d['eeg'][test] / scales['eeg_scale'][None, :, None], device=device, dtype=torch.float32)
            meg = torch.as_tensor(d['meg'][test] / scales['meg_scale'][None, :, None], device=device, dtype=torch.float32)
            with torch.no_grad():
                clean = model.forward_with_details(eeg, meg)
            old = npz(fit / 'predictions.npz')
            assert np.array_equal(old['indices'], test)
            replay = float(np.max(np.abs(clean['log_probability'].exp().cpu().numpy() - old['probability'])))
            if replay > 2e-05:
                raise RuntimeError(f'Full replay differs: {sid}/{fold}: {replay}')
            bank = intervention.extract_bank(model, eeg, meg, clean)
            (changed, checks) = intervention.evaluate_fold(model, bank, clean, d['y'][test], sid, fold, REPEATS)
            assert not coverage[test].any()
            coverage[test] = True
            all_prob[test] = changed['probability']
            audit.append({'fold': fold, 'clean_replay_max_abs_difference': replay})
            print(json.dumps({'subject': sid, 'fold': fold, 'status': 'PASS', 'replay_max_abs_difference': replay}), flush=True)
            del model, eeg, meg, clean, bank, changed
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        assert coverage.all()
        clean = npz(RUN / g / base.FULL_VARIANT / 'oof_predictions.npz')['probability']
        assert float(np.max(np.abs(all_prob[:, :, 0] - clean[:, None]))) < 2e-05
        p = OUT / f'{g}_interventions.npz'
        np.savez_compressed(p, probability=all_prob, y=d['y'], block=d['block'], condition_names=np.asarray(CONDITIONS), covered=coverage)
        scores = {}
        y = d['y'][:, None]
        for (j, name) in enumerate(CONDITIONS):
            pp = all_prob[:, :, j]
            scores[name] = {'accuracy_percent': float(100 * np.mean(pp.argmax(-1) == y)), 'cross_entropy': float(-np.log(np.take_along_axis(pp, y[:, :, None], axis=-1)[..., 0].clip(1e-12)).mean())}
        row = {'subject': sid, 'condition_scores': scores, 'fold_audit': audit, 'artifact': str(p)}
        rows.append(row)
        write(OUT / f'{g}_summary.json', row)
        print(json.dumps({'subject': sid, 'status': 'complete', 'accuracy_percent': {name: round(scores[name]['accuracy_percent'], 3) for name in CONDITIONS}}), flush=True)
    mean = {name: {key: float(np.mean([r['condition_scores'][name][key] for r in rows])) for key in ('accuracy_percent', 'cross_entropy')} for name in CONDITIONS}
    summary = {'status': 'PASS', 'subjects': 12, 'repeats': REPEATS, 'mean_subject_scores': mean}
    write(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
if __name__ == '__main__':
    main()
