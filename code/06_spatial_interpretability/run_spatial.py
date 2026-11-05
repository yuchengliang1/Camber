import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from runtime import resolve_device
from pathlib import Path
import csv
import argparse
import hashlib
import json
import sys
import numpy as np
ROOT = Path(str(_PACKAGE))
REFERENCE = Path(str(_PACKAGE))
OUT = _RESULTS / '06_spatial_interpretability'
sys.path.insert(0, str(ROOT))
import channel_analysis as audit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-channel-values', action='store_true')
    args = parser.parse_args()
    channels = json.loads((_PACKAGE / 'data/channel_positions.json').read_text())['channels']
    positions = {m: sorted([r for r in channels if r['modality'] == m], key=lambda r: r['array_channel_index0']) for m in ('eeg', 'meg')}

    def layout():
        return ([r['template_label'] for r in positions['eeg']], np.array([r['display_xy_head_radius'] if r['has_display_position'] else [np.nan, np.nan] for r in positions['eeg']], float))
    audit.layout = layout
    audit.draw = lambda *args, **kwargs: None
    if not args.reuse_channel_values:
        sys.argv = [str(_PACKAGE / 'code/06_spatial_interpretability/channel_analysis.py'), '--data-root', str(_PACKAGE / 'data'), '--run-root', str(_RESULTS / '01_training_validation/full_and_ablations'), '--out-root', str(OUT), '--subjects', ','.join((str(s) for s in range(29101, 29113))), '--device', str(resolve_device())]
        audit.main()
    with np.load(OUT / 'spatial_audit_arrays.npz', allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files if key not in ('channel_names', 'xy')}
    occ = arrays['test_occlusion'].mean(0)
    power = arrays['power_class'].mean(0)
    for mod in ('eeg', 'meg'):
        arrays[mod + '_channel_names'] = np.asarray([r['template_label'] for r in positions[mod]])
        arrays[mod + '_display_xy'] = np.asarray([r['display_xy_head_radius'] if r['has_display_position'] else [np.nan, np.nan] for r in positions[mod]], dtype=float)
        arrays[mod + '_has_display_position'] = np.asarray([r['has_display_position'] for r in positions[mod]], dtype=bool)
        arrays[mod + '_region_xy'] = np.asarray([r['region_xy_head_radius'] if r['has_display_position'] else [np.nan, np.nan] for r in positions[mod]], dtype=float)
    arrays['array_channel_index0'] = np.arange(64, dtype=np.int64)
    arrays['power_band_hz'] = np.asarray([[8.0, 13.0], [18.0, 30.0]])
    np.savez_compressed(OUT / 'spatial_audit_arrays.npz', **arrays)
    report = json.loads((OUT / 'spatial_audit_report.json').read_text())
    report.pop('central_top8_fraction', None)
    report['maps'] = {f'{m}_occlusion_top10': audit.top(occ[i], [r['template_label'] for r in positions[m]]) for (i, m) in enumerate(('eeg', 'meg'))}
    for (i, mod) in enumerate(('eeg', 'meg')):
        for (bi, band) in enumerate(('mu', 'beta')):
            report['maps'][f'{mod}_{band}_power_top10'] = audit.top(power[i, bi].mean(0), [r['template_label'] for r in positions[mod]])
    report['position_sha256'] = hashlib.sha256((_PACKAGE / 'data/channel_positions.json').read_bytes()).hexdigest()
    report['array_axes'] = {'test_occlusion': ['subject', 'modality', 'channel'], 'power_class': ['subject', 'modality', 'band', 'class', 'channel']}
    (OUT / 'spatial_audit_report.json').write_text(json.dumps(report, indent=2))
    region = OUT / 'common_orientation_projection_assumed/matched_null64'
    region.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in ('eeg', 'meg'):
        for r in positions[m]:
            xy = r['region_xy_head_radius']
            valid = xy is not None and all((v is not None and np.isfinite(v) for v in xy))
            rows.append({'modality': m, 'raw_channel_index0': r['array_channel_index0'], 'template_channel_index0': r['template_channel_index0'], 'template_label': r['template_label'], 'has_position': valid, 'projected_x_right': xy[0] if valid else '', 'projected_y_anterior': xy[1] if valid else ''})
    with (region / 'site_scores.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    scores = []
    for (mi, mod) in enumerate(('eeg', 'meg')):
        for (channel, position) in enumerate(positions[mod]):
            row = {'modality': mod, 'array_channel_index0': channel, 'channel_name': position['template_label'], 'has_position': position['has_display_position'], 'test_occlusion': float(occ[mi, channel])}
            for (bi, band) in enumerate(('mu', 'beta')):
                values = power[mi, bi, :, channel]
                row[band + '_class_range_db'] = float(values.max() - values.min())
                for (label, value) in enumerate(values):
                    row[f'{band}_class_{label}_relative_power_db'] = float(value)
            scores.append(row)
    with (OUT / 'channel_spatial_scores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    print(json.dumps({'status': 'PASS', 'spatial_root': str(OUT)}), flush=True)
if __name__ == '__main__':
    main()
