class hparams:
    sr = 16000
    n_fft = 512
    num_freq = (n_fft // 2) + 1
    n_mels = 40
    n_mfcc = 13
    preemphasis=0.97

    frame_length = 25 # milliseconds
    frame_shift = 10 # milliseconds
    win_length = int(frame_length / 1000 * sr)
    hop_length = int(frame_shift / 1000 * sr)

    min_level_db = -100
    ref_level_db = 20
    max_abs_value = 4.

    classes = ('anger', 'happy', 'neutral', 'sad')
    num_classes = len(classes)

    batch_size = 8
    h_size = 64
    lr = 0.001
    dropout_keep_prob = 0.5

    num_epochs = 100
