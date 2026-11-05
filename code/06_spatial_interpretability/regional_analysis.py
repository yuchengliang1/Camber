import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from runtime import resolve_device
from pathlib import Path
import csv, json, sys, importlib.util, time, hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from frozen_spatial_inference import frozen_forward
torch.set_num_threads(2)
BASE = Path(str(_PACKAGE))
DATA = _PACKAGE / 'data'
RUN = _RESULTS / '01_training_validation/full_and_ablations'
OUT = _RESULTS / '06_spatial_interpretability' / 'common_orientation_projection_assumed' / 'matched_null64'
SITE = OUT / 'site_scores.csv'
sys.path.insert(0, str(Path(str(_PACKAGE / 'code/00_shared/models'))))
import hierarchical_pair_anchor_models_v25 as base
mm = base
config = json.loads((RUN / 'experiment_manifest.json').read_text())['config']
device = resolve_device()
position_hash = hashlib.sha256((_PACKAGE / 'data/channel_positions.json').read_bytes()).hexdigest()
site = list(csv.DictReader(open(SITE)))
REGIONS = ['Anterior', 'Central AP', 'Posterior', 'Left', 'Midline', 'Right']
rng = np.random.default_rng(20261025)
null_draws = 64
mod_index = {'eeg': 0, 'meg': 1}
regions = {}
active = {}
counts = {}
for mod in ('eeg', 'meg'):
    rows = [r for r in site if r['modality'] == mod and r['has_position'] == 'True']
    idx = np.array([int(r['raw_channel_index0']) for r in rows], int)
    x = np.array([float(r['projected_x_right']) for r in rows])
    y = np.array([float(r['projected_y_anterior']) for r in rows])
    active[mod] = idx
    masks = [y > 0.33, (y >= -0.33) & (y <= 0.33), y < -0.33, x < -0.33, (x >= -0.33) & (x <= 0.33), x > 0.33]
    for (name, mask) in zip(REGIONS, masks):
        regions[mod, name] = idx[mask]
        counts[mod, name] = int(mask.sum())

def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}

def eval_patterns(model, xe, xm, yy, base_pred, base_lp, true_base_lp, patterns, batch=5):
    n = len(yy)
    output = []
    for start in range(0, len(patterns), batch):
        group = patterns[start:start + batch]
        k = len(group)
        e = xe.repeat((k, 1, 1))
        m = xm.repeat((k, 1, 1))
        for (j, (mod, chs, _, _, _)) in enumerate(group):
            if mod == 'eeg':
                e[j * n:(j + 1) * n, chs, :] = 0
            else:
                m[j * n:(j + 1) * n, chs, :] = 0
        out = frozen_forward(model, e, m).reshape(k, n, 3)
        p = out.argmax(2)
        pred_lp = out.gather(2, base_pred[None, :, None].expand(k, -1, -1)).squeeze(2)
        true_lp = out.gather(2, yy[None, :, None].expand(k, -1, -1)).squeeze(2)
        for (j, (mod, chs, name, kind, draw)) in enumerate(group):
            output.append({'modality': mod, 'region': name, 'kind': kind, 'draw': draw, 'channel_count': len(chs), 'predicted_logprob_drop': float((base_lp - pred_lp[j]).mean().item()), 'cross_entropy_increase': float((-true_lp[j] - true_base_lp).mean().item()), 'accuracy_drop': float(((base_pred == yy).float() - (p[j] == yy).float()).mean().item())})
        del e, m, out, p, pred_lp, true_lp
    return output
rows = []
baseline_checks = []
cache_check = None
started = time.time()
membership = [{'modality': mod, 'region': name, 'channel_indices0': regions[mod, name].tolist(), 'channel_count': counts[mod, name]} for mod in ('eeg', 'meg') for name in REGIONS]
(OUT / 'regional_channel_membership.json').write_text(json.dumps({'position_sha256': position_hash, 'regions': membership}, indent=2) + '\n')
for sid in range(29101, 29113):
    group = f'sub{sid - 29100:02d}'
    d = npz(DATA / group / 'paired_mi.npz')
    folds = json.loads((_PACKAGE / 'data' / group / 'split.json').read_text())['folds']
    for (fi, fold) in enumerate(folds):
        fit = RUN / group / base.FULL_VARIANT / f'fold_{fi}' / 'outer'
        scale = npz(fit / 'scalers.npz')
        state = torch.load(fit / 'model.pt', map_location='cpu', weights_only=False)
        model = mm.make_model(base.FULL_VARIANT, dropout=config['dropout'])
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        test = np.asarray(fold['test_indices'], int)
        xe = torch.as_tensor(d['eeg'][test] / scale['eeg_scale'][None, :, None], dtype=torch.float32, device=device)
        xm = torch.as_tensor(d['meg'][test] / scale['meg_scale'][None, :, None], dtype=torch.float32, device=device)
        yy = torch.as_tensor(d['y'][test], dtype=torch.long, device=device)
        with torch.no_grad():
            orig = frozen_forward(model, xe, xm)
            base_pred = orig.argmax(1)
            base_lp = orig.gather(1, base_pred[:, None])[:, 0]
            true_base_lp = -orig.gather(1, yy[:, None])[:, 0]
            base_acc = float((base_pred == yy).float().mean().item())
        saved = npz(fit / 'predictions.npz')
        probability = orig.exp().detach().cpu().numpy()
        assert np.array_equal(saved['indices'], test)
        assert np.array_equal(saved['y'], d['y'][test])
        assert np.array_equal(saved['prediction'], base_pred.detach().cpu().numpy())
        error = float(np.max(np.abs(probability - saved['probability'])))
        assert error < 3e-05, error
        baseline_checks.append({'subject': sid, 'fold': fi, 'probability_max_abs_difference': error, 'accuracy': base_acc})
        if cache_check is None:
            pairs = [(xe[:3], xm[:3])]
            e = xe[:3].clone()
            e[:, regions['eeg', 'Central AP'], :] = 0
            pairs.append((e, xm[:3]))
            m = xm[:3].clone()
            m[:, regions['meg', 'Anterior'], :] = 0
            pairs.append((xe[:3], m))
            differences = []
            for (e, m) in pairs:
                with torch.no_grad():
                    direct = model(e, m)
                cached = frozen_forward(model, e, m)
                differences.append(float((direct - cached).abs().max().item()))
                torch.testing.assert_close(direct, cached, rtol=1e-06, atol=2e-06)
            cache_check = {'conditions': ['baseline', 'EEG central occlusion', 'MEG anterior occlusion'], 'max_abs_difference': max(differences)}
        patterns = []
        for mod in ('eeg', 'meg'):
            for name in REGIONS:
                ch = regions[mod, name]
                patterns.append((mod, ch, name, 'region', -1))
                for draw in range(null_draws):
                    rnd = rng.choice(active[mod], size=len(ch), replace=False)
                    patterns.append((mod, rnd, name, 'matched_random', draw))
        result = eval_patterns(model, xe, xm, yy, base_pred, base_lp, true_base_lp, patterns)
        for r in result:
            r.update(subject=sid, fold=fi, baseline_accuracy=base_acc)
            rows.append(r)
        with (OUT / 'regional_occlusion_progress.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (OUT / 'progress.json').write_text(json.dumps({'status': 'RUNNING', 'completed_folds': len(baseline_checks), 'total_folds': 60, 'position_sha256': position_hash, 'seconds': time.time() - started}))
        print(json.dumps({'subject': sid, 'fold': fi, 'completed_folds': len(baseline_checks), 'seconds_elapsed': round(time.time() - started, 1)}), flush=True)
        del model, state, xe, xm, orig, base_pred, base_lp, yy
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print(json.dumps({'subject': sid, 'folds': len(folds), 'seconds_elapsed': round(time.time() - started, 1)}), flush=True)
with open(OUT / 'regional_occlusion_fold.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
metrics = ['predicted_logprob_drop', 'cross_entropy_increase', 'accuracy_drop']
summary = []
for mod in ('eeg', 'meg'):
    for name in REGIONS:
        rr = [r for r in rows if r['modality'] == mod and r['region'] == name]
        item = {'modality': mod, 'region': name, 'channel_count': counts[mod, name]}
        for metric in metrics:
            subj = np.array([[np.mean([r[metric] for r in rr if r['subject'] == sid and r['kind'] == kind]) for kind in ('region', 'matched_random')] for sid in range(29101, 29113)])
            (actual, random) = (subj[:, 0], subj[:, 1])
            excess = actual - random
            boots = rng.choice(len(excess), size=(4000, len(excess)), replace=True)
            ci = np.quantile(excess[boots].mean(1), [0.025, 0.975])
            item[metric + '_actual'] = float(actual.mean())
            item[metric + '_matched_random'] = float(random.mean())
            item[metric + '_excess'] = float(excess.mean())
            item[metric + '_excess_ci_low'] = float(ci[0])
            item[metric + '_excess_ci_high'] = float(ci[1])
            item[metric + '_subjects_positive'] = int((excess > 0).sum())
        summary.append(item)
with open(OUT / 'regional_occlusion_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)
(fig, axes) = plt.subplots(2, 2, figsize=(13.5, 8), sharex='col')
colors = {'eeg': '#2166ac', 'meg': '#e36b2c'}
for (mi, mod) in enumerate(('eeg', 'meg')):
    ss = [r for r in summary if r['modality'] == mod]
    for (ci, idxs) in enumerate((range(3), range(3, 6))):
        ax = axes[mi, ci]
        subset = [ss[i] for i in idxs]
        pos = np.arange(3)
        actual = np.array([r['predicted_logprob_drop_actual'] for r in subset])
        control = np.array([r['predicted_logprob_drop_matched_random'] for r in subset])
        exc = np.array([r['predicted_logprob_drop_excess'] for r in subset])
        lo = np.array([r['predicted_logprob_drop_excess_ci_low'] for r in subset])
        hi = np.array([r['predicted_logprob_drop_excess_ci_high'] for r in subset])
        ax.bar(pos - 0.18, actual, 0.35, color=colors[mod], alpha=0.85, label='Named region')
        ax.bar(pos + 0.18, control, 0.35, color='#aeb6be', label='Matched random')
        ax.axhline(0, color='#555', lw=0.7)
        ax.set_xticks(pos, [f"{r['region']}\n(n={r['channel_count']})" for r in subset], fontsize=9)
        ax.set_title(f"{mod.upper()} | {('front to back' if ci == 0 else 'left to right')}")
        if ci == 0:
            ax.set_ylabel('Predicted-class log-probability drop')
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
        for j in range(3):
            ax.text(j, max(actual[j], control[j]) + max(0.02, abs(actual[j]) * 0.08), f'excess {exc[j]:+.2f}\n[{lo[j]:+.2f}, {hi[j]:+.2f}]', ha='center', va='bottom', fontsize=7.7)
        ymin = min(0, actual.min(), control.min()) - 0.05
        ymax = max(actual.max(), control.max()) * 1.45 + 0.06
        ax.set_ylim(ymin, ymax)
axes[0, 0].legend(loc='upper left', frameon=False, ncol=2, fontsize=8)
fig.suptitle('Frozen Full model: regional channel occlusion on held-out trials', fontsize=14)
fig.text(0.5, 0.015, '95% bootstrap CI for region minus size-matched random control, resampling 12 subjects; 5 folds per subject.', ha='center', fontsize=9)
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(OUT / 'regional_occlusion.png', dpi=220)
fig.savefig(OUT / 'regional_occlusion.svg')
plt.close(fig)
report = {'status': 'PASS', 'dataset': 'dataset', 'model': 'Full', 'subjects': 12, 'folds_per_subject': 5, 'null_draws_per_region_per_fold': null_draws, 'device': str(device), 'metric': 'mean original-predicted-class log probability drop, on held-out trials'}
report.update(position_sha256=position_hash, baseline_checks=baseline_checks, filter_cache_equivalence=cache_check, completed_folds=60)
(OUT / 'regional_occlusion_report.json').write_text(json.dumps(report, indent=2))
(OUT / 'progress.json').write_text(json.dumps({'status': 'PASS', 'completed_folds': 60, 'total_folds': 60, 'position_sha256': position_hash, 'seconds': time.time() - started}))
(OUT / 'regional_occlusion_progress.csv').unlink(missing_ok=True)
print(json.dumps({'status': 'PASS', 'seconds': round(time.time() - started, 1), 'output': str(OUT)}, indent=2), flush=True)
