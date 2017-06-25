import glob
import os
import numpy as np
import pickle


dataset_name = 'fake'

# Change this to a list of the time slices 
files = glob.glob('train/*.npy')


dat_stats={}
dat_stats['name'] = dataset_name
dat_stats['T_bins'] = ['all']
dat_stats['prefix'] = 0

T = len(dat_stats['T_bins'])

def count_words(split):
    dat_stats[split] = np.zeros(T)
    for t, i in enumerate(dat_stats['T_bins']):
        if split=='train':
            print(i)
        for fname in files:
            dat = np.load(fname)
            dat_stats[split][t] += len(dat)

count_words('train')

pickle.dump(dat_stats, open('dat_stats.pkl', "w" ) )
