import matplotlib 
matplotlib.use('Agg')

import pyaudio
import wave
import audioop
from collections import deque
import os
import time
import math
import numpy as np
import librosa

from model import Model
import utils


# Microphone stream config.
CHUNK = 1024  # CHUNKS of bytes to read each time from mic
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 1750  # The threshold intensity that defines silence
                  # and noise signal (an int. lower than THRESHOLD is silence).

SILENCE_LIMIT = 1.5  # Silence limit in seconds. The max ammount of seconds where
                   # only silence is recorded. When this time passes the
                   # recording finishes and the file is delivered.

PREV_AUDIO = 0.5  # Previous audio (in seconds) to prepend. When noise
                  # is detected, how much of previously recorded audio is
                  # prepended. This helps to prevent chopping the beggining
                  # of the phrase.
WAVE_OUTPUT_FILENAME = "samples/output-{}.wav"



def listen_for_speech():
    """
    Listens to Microphone, extracts phrases from it and sends it to 
    Google's TTS service and returns response. a "phrase" is sound 
    surrounded by silence (according to threshold). num_phrases controls
    how many phrases to process before finishing the listening process 
    (-1 for infinite). 
    """

    infer_model = Model('infer', logdir=None)
    infer_model.load_model('logdir/18-07-02T18-30-25/model.ckpt-444')

    #Open stream
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* Listening mic. ")
    audio2send = []
    rel = RATE//CHUNK
    slid_win = deque(maxlen=int(SILENCE_LIMIT*rel))
    #Prepend audio from 0.5 seconds before noise was detected
    prev_audio = deque(maxlen=int(PREV_AUDIO*rel)) 
    started = False
    n = 0
    response = []


    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        slid_win.append(math.sqrt(abs(audioop.avg(data, 4))))
        #print slid_win[-1]
        if(sum([x > THRESHOLD for x in slid_win]) > 0):
            if(not started):
                print()
                print("Starting record of phrase")
                started = True
            audio2send.append(data)
        elif (started is True):
            # The limit was reached, finish capture and deliver.
            wf = save_speech(list(prev_audio) + audio2send, n, p)

            wav, _ = librosa.load(wf, sr=RATE)
            wav = utils.preemphasis(wav)
            S = utils.spectrogram(wav)
            pred = infer_model.infer(np.array([S.T]))
            print(utils.int2emo[pred[0]])

            started = False
            slid_win.clear()
            prev_audio.clear()
            audio2send = []
            n += 1
            print( "Listening ...")
            print()
        else:
            prev_audio.append(data)

    print( "* Done recording")
    stream.close()
    p.terminate()

    return response


def save_speech(data, n, p):
    file = WAVE_OUTPUT_FILENAME.format(n)
    wf = wave.open(file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(data))
    wf.close()
    return file


if(__name__ == '__main__'):
    if not os.path.exists('samples'): os.mkdir('samples')
    listen_for_speech()