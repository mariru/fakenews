import glob
import numpy as np
import pandas as pd
import pickle
import re
import json


df = pd.read_csv('unigram.txt', delimiter='\t',header=None)
labels = df[0].values

vocab = {}

for i,label in enumerate(labels):
    vocab[label] = i

V = 15000

unigram_cnt = df[1].values
phrase_rank = np.array([word.count('_') for word in df[0]])
def words2phrases(text):
    for rank in range(max(phrase_rank),0,-1):
        rank_idx = np.where(phrase_rank == rank)[0]
        for idx in rank_idx:
            phrase = ' '+labels[idx]+' '
            old_phrase = phrase.replace('_',' ')
            text = text.replace(old_phrase, phrase)
    return text

dictionary = {}
count = {}

with open('fulltext.json') as json_file:
    lines = json_file.readlines()

for f_number, line in enumerate(lines):
    text = json.loads(line)['fullText']

    words = re.sub(r'[^a-zA-Z ]',r' ', text.replace('-\n','').replace('\n',' ')).lower()
    words = words2phrases(words).split()
    #words = words.split()
    data = np.zeros(len(words)) + 2*V
    for idx, word in enumerate(words):
        if word in vocab:
            data[idx] = vocab[word]
    data = data.astype('int32')
    data = data[data < V].astype('int32')
    np.save('train/'+str(f_number)+'.npy', data)

