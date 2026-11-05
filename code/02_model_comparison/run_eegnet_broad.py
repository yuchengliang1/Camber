from pathlib import Path
import os
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PACKAGE, RESULTS

def main():
    environment = dict(os.environ)
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    output = RESULTS / '02_model_comparison/eegnet_broad'
    output.mkdir(parents=True, exist_ok=True)
    for subject in range(29101, 29113):
        group = f'sub{subject - 29100:02d}'
        result = output / group
        if (result / 'summary.json').is_file():
            continue
        subprocess.run([sys.executable, '-B', str(PACKAGE / 'code/02_model_comparison/eegnet/run_nested_eegnet.py'), '--data', str(PACKAGE / 'data' / group / 'paired_mi.npz'), '--subject', str(subject), '--out-dir', str(result), '--seed', '27029001', '--device', environment.get('EEG_MEG_DEVICE', 'auto')], check=True, env=environment)
if __name__ == '__main__':
    main()
