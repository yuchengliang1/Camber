import torch

def frozen_forward(model, eeg, meg):
    bank = model.coordinates
    original = bank._band_limited_values
    cache = {}

    def filtered(values):
        key = (values.data_ptr(), tuple(values.shape), values.dtype, values.device)
        if key not in cache:
            cache[key] = original(values)
        return cache[key]
    bank._band_limited_values = filtered
    try:
        with torch.inference_mode():
            return model(eeg, meg)
    finally:
        bank._band_limited_values = original
        cache.clear()
