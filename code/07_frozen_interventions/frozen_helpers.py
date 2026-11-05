import hashlib
import numpy as np

def array_sha(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(str(x.dtype).encode())
    h.update(np.asarray(x.shape, dtype='<i8').tobytes())
    h.update(x.tobytes())
    return h.hexdigest()
