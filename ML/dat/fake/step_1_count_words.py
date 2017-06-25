import glob
import numpy as np
import pandas as pd
import pickle
import re


df = pd.read_csv('phrase_unigram.txt', delimiter='\t',header=None)
labels = df[0].values
V = 15000

f = pd.read_csv('fake.csv')
texts = f.text.values


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

for f_number, text in enumerate(texts):
    if isinstance(text, basestring):
        words = re.sub(r'[^a-zA-Z ]',r' ', text.replace('-\n','').replace('\n',' ')).lower()
        words = words2phrases(words).split()
        #words = words.split()
        data = np.zeros(len(words))
        for idx, word in enumerate(words):
            if word not in dictionary:
                count[len(dictionary)] = 1
                dictionary[word] = len(dictionary)
            data[idx] = dictionary[word]
            count[data[idx]] += 1
        data = data.astype('int32')
        np.save('raw/'+str(f_number)+'.npy', data)

pickle.dump( dictionary, open( 'dict.pkl', "a+" ) )
pickle.dump( count, open( 'counts.pkl', "a+" ) )

df = pd.DataFrame.from_dict(dictionary,orient='index')
cf = pd.DataFrame.from_dict(count,orient='index')
df.columns = ['idx']
cf.columns = ['cnt']
uni = df.join(cf, on = 'idx')
unig = uni.sort_values(by='cnt', ascending = False).reset_index().reset_index()

unig.columns = ['new_idx', 'word', 'old_idx', 'cnt']


old_idx = unig.old_idx.values

unig.head(V).to_csv('unigram.txt',header=False, index = False, sep = '\t', columns = ['word', 'new_idx', 'cnt'])


files = glob.glob('raw/*.npy')
for fname in files:
    print(fname)
    dat = np.load(fname)
    new_dat = np.zeros_like(dat) + 2*V

    for ni, oi in enumerate(old_idx[:V]):
        new_dat[dat == oi] = ni
    new_dat = new_dat[new_dat < V].astype('int32')
    np.save(fname.replace('raw','train'), new_dat)

