A Bidirectional LSTM Network for Audio Classification. Currently setup to be used on the IEMOCAP database for speech emotion recognition, but can be easily modified for other audio classification tasks. Requirements are Tensorflow, Librosa, PyAudio, and tqdm.Begin by installing the requirements.

    pip install -r requirements.txt

## IEMOCAP

To access the IEMOCAP database, submit a request at https://sail.usc.edu/iemocap/release_form.php

When you download the files, unzip them and place them into a directory under data/iemocap/

The directory structure should look like this:

tensorflow-audio-classifier
├── data
    └── iemocap
        ├── Session1
        ├── Session2
        ├── Session3
        ├── Session4
        └── Session5

To train the model, simply execute

    python train.py

A folder will be created within the logdir directory with the saved checkpoints and summary file.

You can stream predictions from a live audio input. On line 48 of serve.py, point load_model to your newly created checkpoint. Then execute,

    python serve.py 

This will create an audio stream with PyAudio that will print the emotion prediction everytime you speak.

