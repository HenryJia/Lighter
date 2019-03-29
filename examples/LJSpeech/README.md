# LJ Speech WaveNet

This directory contains a WaveNet implementation designed to train and run on the LJ Speech dataset

## Training
To train the vanilla WaveNet simply run
``` python wavenet_train.py --csv_dir [LJSpeech metadata.csv location] --audio_dir [LJSpeech wav file directory] <additional arguments>```

For additional arguments see Line 30 of wavenet_train.py (I'm too lazy to write them all out again here). You can also specify arguments via json using ``` --json config.json``` and I provide a json file of the configuration I used

## Generation
To generate audio simply run. I provide pretrained weights in this repository via git LFS
``` python wavenet_run.py --csv_dir [LJSpeech metadata.csv location] --audio_dir [LJSpeech wav file directory] <additional arguments>``` You can also specify arguments via json using ``` --json config.json``` and I provide a json file of the configuration I used. Note you should train and generate using the same arguments/json configuration. Note since WaveNet is **NOT** a full text to speech model, we need the Mel spectrograms so we still need to dataset to try out generation.

For additional arguments see Line 30 of wavenet_run.py (I'm too lazy to write them all out again here)

## Other files in this directory
```teacher_wavenet_train.py``` and ```teacher_wavenet_generate.py``` are implementations of the ClariNet architecture which I am still developing. They work, but they don't give any good results at the moment, so just ignore them for now
``` data_permutations.npy``` specifies which files were used for training and validation so when we're generating we don't end up using the training set.
