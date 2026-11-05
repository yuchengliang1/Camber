import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
import json
import numpy as np

def make_splits(block, y):
    folds = []
    for (fold, test_block) in enumerate(sorted(np.unique(block).tolist())):
        train = np.flatnonzero(block != test_block)
        test = np.flatnonzero(block == test_block)
        inner = []
        for (j, validation_block) in enumerate(sorted(set(block.tolist()) - {test_block})):
            tr = np.flatnonzero((block != test_block) & (block != validation_block))
            va = np.flatnonzero(block == validation_block)
            assert len(tr) == 90 and len(va) == 30
            assert not (set(tr) & set(va) or set(tr) & set(test) or set(va) & set(test))
            inner.append({'inner_fold': j, 'validation_block': int(validation_block), 'train_indices': tr.tolist(), 'validation_indices': va.tolist(), 'train_label_counts': np.bincount(y[tr], minlength=3).tolist(), 'validation_label_counts': np.bincount(y[va], minlength=3).tolist()})
        assert len(train) == 120 and len(test) == 30
        folds.append({'fold': fold, 'held_out_block': int(test_block), 'train_indices': train.tolist(), 'test_indices': test.tolist(), 'train_label_counts': np.bincount(y[train], minlength=3).tolist(), 'test_label_counts': np.bincount(y[test], minlength=3).tolist(), 'inner': inner})
    assert sorted((i for f in folds for i in f['test_indices'])) == list(range(150))
    return {'folds': folds}
if __name__ == '__main__':
    for index in range(1, 13):
        folder = _PACKAGE / 'data' / f'sub{index:02d}'
        with np.load(folder / 'paired_mi.npz', allow_pickle=False) as data:
            result = make_splits(data['block'], data['y'])
        text = json.dumps(result, indent=2) + '\n'
        split_path = folder / 'split.json'
        if split_path.exists():
            previous = json.loads(split_path.read_text())
            if result['folds'] != previous['folds']:
                raise RuntimeError(f'Existing split does not match block-derived split: {split_path}')
        else:
            split_path.write_text(text, encoding='utf-8')
        target = _RESULTS / '01_training_validation/splits' / folder.name / 'split.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
