import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from runtime import resolve_device
from pathlib import Path
import argparse, json, sys, csv, re, hashlib, importlib.util
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
CODE = Path(str(_PACKAGE / 'code/00_shared/models'))
sys.path.insert(0, str(CODE))
import hierarchical_pair_anchor_models_v25 as base_model
FULL = base_model.FULL_VARIANT
FS = 250.0
BANDS = (('mu', 8.0, 13.0), ('beta', 18.0, 30.0))

def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}

def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')

def layout():
    rows = json.loads((_PACKAGE / 'data/channel_positions.json').read_text())['channels']
    rows = sorted([r for r in rows if r['modality'] == 'eeg'], key=lambda r: r['array_channel_index0'])
    return ([r['template_label'] for r in rows], np.asarray([r['display_xy_head_radius'] if r['has_display_position'] else [np.nan, np.nan] for r in rows], float))

def load_model(path, mm, device, dropout):
    model = mm.make_model(FULL, dropout=float(dropout))
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def occlusion(model, xe, xm, mod, channels, batch=8):
    with torch.no_grad():
        base = model(xe, xm)
        cls = base.argmax(1)
        basev = base.gather(1, cls[:, None])[:, 0]
    out = np.zeros((xe.shape[0], len(channels)), np.float32)
    n = xe.shape[0]
    for start in range(0, len(channels), batch):
        ch = np.asarray(channels[start:start + batch], int)
        k = len(ch)
        er = xe.repeat_interleave(k, 0)
        mr = xm.repeat_interleave(k, 0)
        target = er if mod == 'eeg' else mr
        for (j, c) in enumerate(ch):
            target[j::k, int(c), :] = 0.0
        with torch.no_grad():
            p = model(er, mr)
            cr = cls.repeat_interleave(k)
            val = p.gather(1, cr[:, None])[:, 0]
        out[:, start:start + k] = (basev.repeat_interleave(k) - val).reshape(n, k).detach().cpu().numpy()
        del er, mr, p
    return out

def band_power(x):
    f = np.fft.rfftfreq(x.shape[-1], 1 / FS)
    spec = np.fft.rfft(x.astype(np.float64), axis=-1)
    total = np.mean(np.abs(spec[..., (f >= 1) & (f <= 45)]) ** 2, axis=-1)
    out = np.zeros((2, x.shape[0], x.shape[1]), np.float32)
    for (bi, (_, lo, hi)) in enumerate(BANDS):
        bp = np.mean(np.abs(spec[..., (f >= lo) & (f <= hi)]) ** 2, axis=-1)
        out[bi] = 10 * np.log10(np.maximum(bp, 1e-20) / np.maximum(total, 1e-20))
    return out

def top(values, names):
    ix = np.argsort(-values)[:10]
    return [{'index0': int(i), 'name': names[int(i)], 'value': float(values[int(i)])} for i in ix]

def draw(out, names, xy, maps):
    (fig, ax) = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    specs = [('eeg_occ', 'Full EEG occlusion'), ('meg_occ', 'Full MEG occlusion'), ('eeg_mu', 'EEG task mu relative power'), ('meg_mu', 'MEG task mu relative power'), ('eeg_beta', 'EEG task beta relative power'), ('meg_beta', 'MEG task beta relative power')]
    for (a, (key, title)) in zip(ax.flat, specs):
        v = maps[key]
        lim = max(float(np.max(np.abs(v))), 1e-08)
        sc = a.scatter(xy[:, 0], xy[:, 1], c=v, s=62, cmap='coolwarm', vmin=-lim, vmax=lim, edgecolors='k', linewidths=0.2)
        for (i, n) in enumerate(names):
            if n in ('C3', 'Cz', 'C4'):
                a.text(xy[i, 0], xy[i, 1], n, fontsize=7, ha='center', va='bottom')
        a.set_title(title, fontsize=9)
        a.set_aspect('equal')
        a.set_xticks([])
        a.set_yticks([])
        fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02)
    fig.suptitle('Full spatial analysis', fontsize=13)
    fig.savefig(out / 'global12_spatial_maps.png', dpi=220)
    fig.savefig(out / 'global12_spatial_maps.svg')
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--run-root', type=Path, required=True)
    ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--subjects', required=True)
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    a.out_root.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(a.device)
    mm = base_model
    (names, xy) = layout()
    subjects = [int(x) for x in a.subjects.split(',')]
    all_occ_test = []
    all_occ_train = []
    all_power = []
    acc = []
    for s in subjects:
        g = f'sub{s - 29100:02d}'
        d = load_npz(a.data_root / g / 'paired_mi.npz')
        split = json.loads((_PACKAGE / 'data' / g / 'split.json').read_text())
        cfg = json.loads((a.run_root / 'experiment_manifest.json').read_text())['config']
        occ_te = np.zeros((2, 64), np.float32)
        occ_tr = np.zeros((2, 64), np.float32)
        for (f, fold) in enumerate(split['folds']):
            fit = a.run_root / g / FULL / f'fold_{f}' / 'outer'
            sc = load_npz(fit / 'scalers.npz')
            es = sc['eeg_scale']
            ms = sc['meg_scale']
            model = load_model(fit / 'model.pt', mm, dev, cfg['dropout'])
            tr = np.asarray(fold['train_indices'], np.int64)
            te = np.asarray(fold['test_indices'], np.int64)
            xe_tr = torch.as_tensor(d['eeg'][tr] / es[None, :, None], dtype=torch.float32, device=dev)
            xm_tr = torch.as_tensor(d['meg'][tr] / ms[None, :, None], dtype=torch.float32, device=dev)
            xe_te = torch.as_tensor(d['eeg'][te] / es[None, :, None], dtype=torch.float32, device=dev)
            xm_te = torch.as_tensor(d['meg'][te] / ms[None, :, None], dtype=torch.float32, device=dev)
            occ_tr[0] += np.maximum(occlusion(model, xe_tr, xm_tr, 'eeg', np.arange(64)).mean(0), 0) / 5
            occ_tr[1] += np.maximum(occlusion(model, xe_tr, xm_tr, 'meg', np.arange(64)).mean(0), 0) / 5
            occ_te[0] += occlusion(model, xe_te, xm_te, 'eeg', np.arange(64)).mean(0) / 5
            occ_te[1] += occlusion(model, xe_te, xm_te, 'meg', np.arange(64)).mean(0) / 5
            with np.load(fit / 'predictions.npz', allow_pickle=False) as z:
                acc.append(float(np.mean(z['prediction'] == d['y'][z['indices']])))
            del model, xe_tr, xm_tr, xe_te, xm_te
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
        power = np.stack([band_power(d['eeg']), band_power(d['meg'])])
        power_class = np.zeros((2, 2, 3, 64), np.float32)
        for mi in range(2):
            for bi in range(2):
                for c in range(3):
                    power_class[mi, bi, c] = power[mi, bi][d['y'] == c].mean(0)
        all_occ_test.append(occ_te)
        all_occ_train.append(occ_tr)
        all_power.append(power_class)
        print(json.dumps({'subject': s, 'status': 'spatial_complete'}), flush=True)
    occ = np.stack(all_occ_test)
    rank = np.stack(all_occ_train)
    power = np.stack(all_power)
    pm = power.mean(0)
    om = occ.mean(0)
    rm = rank.mean(0)
    contrast = pm.max(2) - pm.min(2)
    corr = {}
    perm = np.random.default_rng(20261025).permutation(64)
    for (mi, mod) in enumerate(('eeg', 'meg')):
        corr[mod + '_occlusion_vs_mu'] = float(pearsonr(om[mi], contrast[mi, 0]).statistic)
        corr[mod + '_occlusion_vs_beta'] = float(pearsonr(om[mi], contrast[mi, 1]).statistic)
        corr[mod + '_permutation_control_mu'] = float(pearsonr(om[mi], contrast[mi, 0][perm]).statistic)
        corr[mod + '_permutation_control_beta'] = float(pearsonr(om[mi], contrast[mi, 1][perm]).statistic)
    central = np.array([bool(re.match('^(FC|C|CP)', n)) for n in names])
    maps = {'eeg_occ': om[0], 'meg_occ': om[1], 'eeg_mu': pm[0, 0].mean(0), 'meg_mu': pm[1, 0].mean(0), 'eeg_beta': pm[0, 1].mean(0), 'meg_beta': pm[1, 1].mean(0)}
    np.savez_compressed(a.out_root / 'spatial_audit_arrays.npz', test_occlusion=occ, train_rank=rank, power_class=power, channel_names=np.asarray(names), xy=xy)
    draw(a.out_root, names, xy, maps)
    report = {'status': 'PASS', 'subjects': subjects, 'accuracy_mean_percent': float(100 * np.mean(acc)), 'maps': {'full_eeg_occlusion_top10': top(om[0], names), 'full_meg_occlusion_top10': top(om[1], names), 'eeg_mu_power_top10': top(maps['eeg_mu'], names), 'meg_mu_power_top10': top(maps['meg_mu'], names), 'eeg_beta_power_top10': top(maps['eeg_beta'], names), 'meg_beta_power_top10': top(maps['meg_beta'], names)}, 'central_top8_fraction': {'eeg_occlusion': float(central[np.argsort(-om[0])[:8]].mean()), 'meg_occlusion': float(central[np.argsort(-om[1])[:8]].mean())}, 'correlations': corr, 'controls': {'permutation_seed': 20261025, 'permutation': perm.tolist()}}
    write(a.out_root / 'spatial_audit_report.json', report)
    print(json.dumps({'status': 'PASS', 'out': str(a.out_root)}), flush=True)
if __name__ == '__main__':
    main()
