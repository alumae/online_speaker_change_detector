from collections import OrderedDict, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from scipy.io import wavfile

def tp_fp_fn(preds, targets, tolerance=50):
    """
    Returns a tuple of true positives, false positives and false negatives given
    predictions and target values.
    """    
    preds_idx = np.where(preds)[0]
    targets_idx = np.where(targets)[0]

    n = len(targets_idx)
    m = len(preds_idx)
    
    if (m==0):
        return 0.0, 0.0, n
    elif (n==0):
        return 0.0, m, 0.0

    delta = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            delta[i, j] = abs(targets_idx[i]-preds_idx[j]) 

    delta[np.where(delta > tolerance)] = np.inf

    # h always contains the minimum value in delta matrix
    # h == np.inf means that no boundary can be matched
    h = np.amin(delta)
    n_matches = 0.

    # while there are still boundaries to match
    while h < np.inf:
        # increment match count
        n_matches += 1

        # find boundaries to match
        k = np.argmin(delta)
        i = k // m
        j = k % m

        # make sure they cannot be matched again
        delta[i, :] = np.inf
        delta[:, j] = np.inf

        # update minimum value in delta
        h = np.amin(delta)
    return n_matches, m-n_matches, n-n_matches

def load_wav_file(sound_file_path, sample_rate):
    """Load the wav file at the given file path and return a float32 numpy array."""
    with open(sound_file_path, 'rb') as f:
        wav_sample_rate, sound_np = wavfile.read(f)
        # FIXME: resample is necessary
        assert(sample_rate == wav_sample_rate)

        if sound_np.dtype != np.float32:
            assert sound_np.dtype == np.int16
            sound_np = np.divide(
                sound_np, 32768, dtype=np.float32
            )  # ends up roughly between -1 and 1
        assert(len(sound_np.shape) == 1)
        return sound_np