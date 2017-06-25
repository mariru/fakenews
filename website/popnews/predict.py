import glob
import json
import math
import numpy as np
import os
import pandas as pd
import pickle
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, 'parameters/unigram.txt'),
                 delimiter='\t',header=None)
labels = df[0].values
vocab = {}
for i,label in enumerate(labels):
    vocab[label] = i
V = 15000
unigram_cnt = df[1].values
phrase_rank = np.array([word.count('_') for word in df[0]])

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def neural_network(x, w_1, w_2):
    return np.tanh(x.dot(w_1)).dot(w_2)

def get_model():
    fit = pickle.load(open(os.path.join(BASE_DIR,
                                        'parameters/model_params.pkl'), 'rb'),
                      encoding='latin1')
    return fit['emb'], fit['w_1'], fit['w_2']

def words2phrases(text):
    for rank in range(max(phrase_rank),0,-1):
        rank_idx = np.where(phrase_rank == rank)[0]
        for idx in rank_idx:
            phrase = ' '+labels[idx]+' '
            old_phrase = phrase.replace('_',' ')
            text = text.replace(old_phrase, phrase)
    return text

def text2numbers(text):
    words = re.sub(r'[^a-zA-Z ]',r' ', text.replace('-\n','').replace('\n',' ')).lower()
    words = words2phrases(words).split()
    data = np.zeros(len(words)) + 2*V
    for idx, word in enumerate(words):
        if word in vocab:
            data[idx] = vocab[word]
    data = data.astype('int32')
    data = data[data < V].astype('int32')
    return data

def extract_features(text, emb):
     if len(text) == 0:
         return np.zeros((emb.shape[1]))
     return np.mean(emb[text], axis=0)

def pred_bias(text):
    text = text2numbers(text)
    emb, w_1, w_2 = get_model()
    features = extract_features(text, emb)
    pred = sigmoid(neural_network(features, w_1, w_2))
    # Rescale to use more of 0-1 range

    pred = pred - 0.5

    # Flip because frontend assumes 0 is liberal and 1 is conservative and
    # ML model assumes 0 is conservative and 1 is liberal.
    pred = pred * -5
    pred = pred + 0.5
    return pred
