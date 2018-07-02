import numpy as np
import tensorflow as tf
import re
import os
import glob
import random
import math
import librosa
from scipy import signal
from collections import namedtuple, Counter

from hparams import hparams as hp
import utils


class Helper(namedtuple("Helper", ("initializer", "input_seq", "labels", "num_batches", "weights"))):
    pass


def build_iemocap_generator(mode):
    abbr2emo = { 'ang' : 'anger', 'dis' : 'disgust', 'fea' : 'fear', 'hap' : 'happy', 'neu' : 'neutral', 'sad' : 'sad', 'sur' : 'surprise', 'xxx' : 'xxx', 'fru' : 'frustration', 'exc' : 'excited', 'oth' : 'other' }

    wav_files = glob.glob('data/iemocap/*/sentences/wav/*/*.wav')
    wav_files = { os.path.basename(f).replace('.wav','') : f for f in wav_files }

    data = glob.glob('data/iemocap/*/dialog/EmoEvaluation/*.txt')
    if mode == 'train':
        data = [d for d in data if 'Session5' not in d]
    elif mode == 'test':
        data = [d for d in data if 'Session5' in d]
    data = [open(file, 'r').read().split('\n\n')[1:-1] for file in data]
    data = [block.split('\n')[0].split('\t') for file in data for block in file]
    data = [(d[1], d[2]) for d in data if abbr2emo[d[2]] in hp.classes]

    total = len(data)
    count = Counter([d[1] for d in data])
    weights = { utils.emo2int[abbr2emo[e]] : (1/len(hp.classes))/(count[e]/total) for e in count }
    weights = [w[1] for w in sorted(weights.items())]

    num_batches = math.ceil(len(data) / hp.batch_size)

    def generator():
        for f, emo in data:
            f = wav_files[f]
            wav, _ = librosa.load(f, sr=hp.sr)
            wav = utils.preemphasis(wav)
            S = utils.spectrogram(wav)

            label = utils.emo2int[abbr2emo[emo]]

            yield (S.T, label)

    return generator, num_batches, weights


def build_helper(mode):
    generator, num_batches, weights = build_iemocap_generator(mode)
    ds = tf.data.Dataset.from_generator(generator, (tf.float32, tf.int32))
    ds = ds.padded_batch(hp.batch_size, padded_shapes=(tf.TensorShape([None, hp.num_freq]), tf.TensorShape([])))
    ds = ds.prefetch(1)
    iterator = ds.make_initializable_iterator()
    initializer = iterator.initializer
    input_seq, labels = iterator.get_next()

    return Helper(initializer=initializer, input_seq=input_seq, labels=labels, num_batches=num_batches, weights=weights)

