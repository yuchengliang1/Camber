import os
import platform
import importlib.metadata


def resolve_device(name=None):
    import torch
    requested = name or os.environ.get('EEG_MEG_DEVICE', 'auto')
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: auto, cpu, cuda, cuda:N')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu or --device auto.')
    return device


def runtime_record():
    import torch
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'scipy', 'torch', 'scikit-learn', 'matplotlib')},
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
    }
