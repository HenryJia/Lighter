from time import time
from pathlib import Path
import random, os, warnings
from collections import OrderedDict

import numpy as np

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim

from .core import Transform

from tqdm import tqdm



class Word2Vec(Transform):
    """
    Transform for Word2Vec so we can train Word2Vec on our dataset and use it as a transform

    We do this using nltk and gensim.models.Word2Vec

    Parameters
    ----------
    text_dir: String
        Directory of the text files we want to do this on
    dim: Integer
        Dimensionality of the Word2Vec embeddings
    workers: Integer
        Number of workers to use to train the Word2Vec
    """
    def __init__(self, text_dir, dim = 128, workers = 10):
        self.text_dir = text_dir
        self.dim = dim
        self.workers = workers


    def train(self):
        # Start training the Word2Vec
        # First create a list of files
        self.file_list = [s for s in list(Path(self.text_dir).rglob("*.txt"))]

        # Now read and tokenize
        print('Loading texts')
        self.sentences = ' '.join([Path(fn).read_text() for fn in tqdm(self.file_list)])
        print('Tokenising texts')
        self.sentences = [word_tokenize(sent.lower()) for sent in tqdm(sent_tokenize(self.sentences))] # Lower case everything

        # I've commented this section out because we could usep pretrained stuff but they may not have the vocabulary we need
        #word2vec_sample = str(nltk.data.find('models/word2vec_sample/pruned.word2vec.txt'))
        #self.model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)]

        # Need to set min_count = 1, so we can deal with words we onyl encounter once if we have to
        print('Training Word2Vec model')
        self.model = gensim.models.Word2Vec(self.sentences, size = self.dim, workers = self.workers, min_count = 1)
        print('Done')


    def __call__(self, x): # We expect the input to be a string of sentences
        # Don't need to split it into sentences then into words, tokenising every word is enough
        sentence = word_tokenize(x.lower())

        out = np.zeros((len(sentence), self.dim), dtype = np.float32)
        for i, w in enumerate(sentence):
            out[i] = self.model[w]
        return out


    def save(self, fn):
        self.model.save(fn)


    def load(self, fn):
        self.model = Word2Vec.load(fn)


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'text_dir={0}'.format(self.text_dir)
        format_string += 'dim={0}'.format(self.dim)
        format_string += 'workers={0}'.format(self.workers)
        format_string += ')'
        return format_string



class Char2Vec(Transform):
    def __call__(self, x):
        out = np.zeros((len(x), 256), dtype = np.float32) # Fix dimensionality to 256 for 8bit ASCII

        for i, c in enumerate(x):
            out[i, ord(c)] = 1 # Convert to 1 hot

        return out
