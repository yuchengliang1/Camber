from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from runtime import resolve_device, runtime_record
ROOT = Path(__file__).resolve().parent.parent
STAGES = json.loads((ROOT / 'code/stages.json').read_text())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['list', 'all', 'replay', 'smoke'] + list(STAGES))
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.stage == 'list':
        print('\n'.join(STAGES))
        return
    environment = dict(os.environ)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    device = str(resolve_device(args.device))
    environment['EEG_MEG_DEVICE'] = device
    environment['MPLBACKEND'] = 'Agg'
    results = (args.results_root or ROOT / 'local_runs' / datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    archived = (ROOT / 'results').resolve()
    if results == archived or archived in results.parents:
        parser.error('Use a separate --results-root; results/ contains the saved experiments.')
    results.mkdir(parents=True, exist_ok=True)
    if args.stage in ('replay', 'smoke'):
        subprocess.run([sys.executable, '-B', str(ROOT / 'code/verify_local.py'), args.stage,
                        '--output', str(results), '--device', device], cwd=ROOT,
                       env=environment, check=True)
        return
    environment['EEG_MEG_RESULTS'] = str(results)
    for category in ('01_training_validation', '02_model_comparison', '03_ablation', '04_parameter_sensitivity', '06_spatial_interpretability'):
        (results / category).mkdir(parents=True, exist_ok=True)
    selected = STAGES if args.stage == 'all' else {args.stage: STAGES[args.stage]}
    record_path = results / 'run_status.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else {
        'runtime': runtime_record(), 'device': device, 'stages': {}}
    for (stage, command) in selected.items():
        if args.resume and record['stages'].get(stage, {}).get('status') == 'complete':
            continue
        command = list(command)
        if '--device' in command:
            command[command.index('--device') + 1] = device
        command = [str(results / item[8:]) if item.startswith('results/') else str(ROOT / item) if item.startswith(('code/', 'data/', 'projection/')) else item for item in command]
        item = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(), 'command': command}
        record['stages'][stage] = item
        record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(stage, flush=True)
        try:
            subprocess.run([sys.executable, '-B'] + command, cwd=ROOT, env=environment, check=True)
        except Exception:
            item['status'] = 'failed'
            record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
            raise
        item.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
        record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(str(results), flush=True)
if __name__ == '__main__':
    main()
