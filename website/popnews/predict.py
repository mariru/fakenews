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

def get_model():
    fit = pickle.load(open(os.path.join(BASE_DIR,
                                        'parameters/model_params.pkl'), 'rb'),
                      encoding='latin1')
    return fit['emb'], fit['w']

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
    emb, w = get_model()
    features = extract_features(text, emb)
    pred = sigmoid(features.dot(w))
    return pred
