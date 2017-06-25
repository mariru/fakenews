import glob
import numpy as np
import pandas as pd
import pickle
import re


df = pd.read_csv('unigram.txt', delimiter='\t',header=None)
labels = df[0].values
V = 15000

f = pd.read_csv('fake.csv')
all_sites = f.site_url.values

url = pd.read_csv('site_counts.csv',header=None)

sites = url[0].values
bias = url[2].values

site_bias = {}

for i, site in enumerate(sites):
     site_bias[site] = bias[i]

left = []

right = []

for f_number, site in enumerate(all_sites):
    if site_bias[site] == 'left':
        left = left + ['raw/'+str(f_number)+'.npy']
    if site_bias[site] == 'right':
        right = right + ['raw/'+str(f_number)+'.npy']



with open('left.txt', 'w') as thefile:
  for item in left:
    thefile.write("%s\n" % item)

with open('right.txt', 'w') as thefile:
  for item in right:
    thefile.write("%s\n" % item)

