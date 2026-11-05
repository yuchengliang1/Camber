from pathlib import Path
import argparse
import csv
import json
import os
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
HERE = Path(__file__).resolve().parent
PARAMETER_NAMES = ('angle_degrees', 'scale_x', 'scale_y', 'translation_x', 'translation_y')

def project_native(channels, radius=0.89):
    result = {}
    scales = {}
    for modality in ('eeg', 'meg'):
        rows = sorted((r for r in channels if r['modality'] == modality), key=lambda r: r['template_channel_index0'])
        valid = []
        plane = []
        for row in rows:
            xyz = np.asarray(row['native_position']['xyz'], dtype=float)
            allowed = row['position_status'] == ('standard_1020_template' if modality == 'eeg' else 'bst_native')
            if not allowed or not np.isfinite(xyz).all():
                continue
            unit = row['native_position']['unit']
            if unit not in ('m', 'mm'):
                raise ValueError(unit)
            xyz_mm = xyz * (1000.0 if unit == 'm' else 1.0)
            point = xyz_mm[:2] if modality == 'eeg' else np.array([-xyz_mm[1], xyz_mm[0]])
            valid.append(row['array_channel_index0'])
            plane.append(point)
        plane = np.asarray(plane, dtype=float)
        maximum_radius = np.linalg.norm(plane, axis=1).max()
        if not np.isfinite(maximum_radius) or maximum_radius <= 0:
            raise ValueError(modality)
        factor = radius / maximum_radius
        result[modality] = {'indices': valid, 'xy': plane * factor}
        scales[modality] = {'native_plane_max_radius_mm': float(maximum_radius), 'normalization_factor_per_mm': float(factor)}
    return (result, scales)

def apply_transform(points, parameters):
    (angle, sx, sy, tx, ty) = parameters
    (c, s) = (np.cos(angle), np.sin(angle))
    rotation = np.array([[c, -s], [s, c]])
    return points @ rotation.T * np.array([sx, sy]) + np.array([tx, ty])

def fit_geometry(eeg_xy, meg_xy):

    def objective(parameters):
        aligned = apply_transform(meg_xy, parameters)
        squared = cdist(aligned, eeg_xy) ** 2
        chamfer = 0.5 * (np.mean(np.min(squared, axis=0)) + np.mean(np.min(squared, axis=1)))
        (angle, sx, sy, tx, ty) = parameters
        regularizer = 0.01 * angle ** 2 + 0.012 * ((sx - 1) ** 2 + (sy - 1) ** 2) + 0.006 * (tx ** 2 + ty ** 2)
        return chamfer + regularizer
    bounds = [(-np.deg2rad(10), np.deg2rad(10)), (0.85, 1.15), (0.85, 1.15), (-0.08, 0.08), (-0.08, 0.08)]
    starts = [np.array([np.deg2rad(angle), 1.0, 1.0, 0, 0]) for angle in (-5, 0, 5)]
    fits = [minimize(objective, start, method='Powell', bounds=bounds, options={'maxiter': 350, 'xtol': 1e-06, 'ftol': 1e-08}) for start in starts]
    best = min(fits, key=lambda fit: fit.fun)
    return (best.x, {'success': bool(best.success), 'objective': float(best.fun), 'iterations': int(best.nit)})

def parameter_array(record):
    values = np.array([record[name] for name in PARAMETER_NAMES], dtype=float)
    values[0] = np.deg2rad(values[0])
    return values

def parameter_record(values):
    values = np.asarray(values).copy()
    values[0] = np.rad2deg(values[0])
    return {name: float(value) for (name, value) in zip(PARAMETER_NAMES, values)}

def reconstruct(channels, radius, parameters):
    (plane, scales) = project_native(channels, radius)
    regions = {(modality, index): xy for (modality, values) in plane.items() for (index, xy) in zip(values['indices'], values['xy'])}
    displayed = {key: point.copy() for (key, point) in regions.items()}
    aligned = apply_transform(plane['meg']['xy'], parameters)
    for (index, point) in zip(plane['meg']['indices'], aligned):
        displayed['meg', index] = point
    output = []
    for row in channels:
        key = (row['modality'], row['array_channel_index0'])
        output.append({**row, 'has_display_position': key in regions, 'region_xy_head_radius': regions[key].tolist() if key in regions else None, 'display_xy_head_radius': displayed[key].tolist() if key in displayed else None})
    return (output, plane, aligned, scales)

def draw(plane, aligned, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    (fig, axes) = plt.subplots(1, 2, figsize=(9.6, 4.8), layout='constrained')
    for (ax, meg_xy, title) in zip(axes, [plane['meg']['xy'], aligned], ['Region coordinates', 'Display coordinates']):
        ax.add_patch(Circle((0, 0), 1, fill=False, color='#333333', linewidth=1))
        ax.plot([-0.1, 0, 0.1], [1, 1.12, 1], color='#333333', linewidth=1)
        for side in (-1, 1):
            ax.plot(np.array([1, 1.07, 1.07, 1]) * side, [0.12, 0.12, -0.12, -0.12], color='#333333', linewidth=1)
        ax.scatter(plane['eeg']['xy'][:, 0], plane['eeg']['xy'][:, 1], s=24, marker='o', facecolors='none', edgecolors='#2166ac', label='EEG')
        ax.scatter(meg_xy[:, 0], meg_xy[:, 1], s=25, marker='^', facecolors='none', edgecolors='#b35806', label='MEG')
        ax.set(xlim=(-1.15, 1.15), ylim=(-1.12, 1.18), aspect='equal', title=title)
        ax.axis('off')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'projection_steps.{extension}', dpi=240)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=HERE.parent / 'data/channel_positions.json')
    parser.add_argument('--transform', type=Path, default=HERE / 'display_transform.json')
    parser.add_argument('--output-dir', type=Path, default=Path(os.environ.get('EEG_MEG_RESULTS', str(HERE.parent / 'results'))) / '06_spatial_interpretability/projection')
    parser.add_argument('--refit', action='store_true')
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding='utf-8'))
    settings = json.loads(args.transform.read_text(encoding='utf-8'))
    parameters = parameter_array(settings)
    optimization = None
    if args.refit:
        (projected, _) = project_native(source['channels'], settings['normalization_radius'])
        (parameters, optimization) = fit_geometry(projected['eeg']['xy'], projected['meg']['xy'])
    (channels, plane, aligned, scales) = reconstruct(source['channels'], settings['normalization_radius'], parameters)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'channel_positions.json').write_text(json.dumps({'index_base': 0, 'channels': channels}, indent=2) + '\n', encoding='utf-8')
    rows = []
    for row in channels:
        region = row['region_xy_head_radius'] or ['', '']
        display = row['display_xy_head_radius'] or ['', '']
        rows.append({'modality': row['modality'], 'array_channel_index0': row['array_channel_index0'], 'template_channel_index0': row['template_channel_index0'], 'template_label': row['template_label'], 'has_position': row['has_display_position'], 'region_x_right': region[0], 'region_y_anterior': region[1], 'display_x_right': display[0], 'display_y_anterior': display[1]})
    with (args.output_dir / 'projected_positions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = np.min(cdist(plane['meg']['xy'], plane['eeg']['xy']), axis=1)
    after = np.min(cdist(aligned, plane['eeg']['xy']), axis=1)
    report = {**parameter_record(parameters), 'normalization_radius': settings['normalization_radius'], 'normalization': scales, 'position_only': True, 'uses_z': False, 'refit': args.refit, 'position_counts': {mod: len(value['indices']) for (mod, value) in plane.items()}, 'nearest_distance_mean_before': float(before.mean()), 'nearest_distance_mean_after': float(after.mean())}
    if optimization is not None:
        report['optimization'] = optimization
    (args.output_dir / 'projection_parameters.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    draw(plane, aligned, args.output_dir)
    print(json.dumps({'status': 'PASS', 'output': str(args.output_dir), 'position_counts': report['position_counts']}))
if __name__ == '__main__':
    main()
