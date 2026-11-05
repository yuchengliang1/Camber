from pathlib import Path
import os
import sys
CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parent
RESULTS = Path(os.environ.get('EEG_MEG_RESULTS', str(PACKAGE / 'results'))).resolve()
sys.dont_write_bytecode = True
for directory in [CODE] + sorted((p for p in CODE.rglob('*') if p.is_dir())):
    sys.path.insert(0, str(directory))

def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'results':
        return RESULTS.joinpath(*path.parts[1:])
    return PACKAGE / path
