import librosa
from scipy import signal
import numpy as np
from hparams import hparams as hp 

# for IEMOCAP
emo2int = { k : i for i, k in enumerate(hp.classes) }
int2emo = { i : k for k, i in emo2int.items() }

def preemphasis(wav):
    return signal.lfilter([1, -hp.preemphasis], [1], wav)

def spectrogram(wav):
    S = _stft(wav)
    S = np.abs(S)
    S = _amp_to_db(S) - hp.ref_level_db
    S = _normalize(S)
    return S

def _amp_to_db(S):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, S))

def _normalize(S):
    return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)

def _stft(wav):
    return librosa.stft(y=wav, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)