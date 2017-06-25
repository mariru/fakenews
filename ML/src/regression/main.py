import glob
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from utils import *
from tensorflow.contrib.distributions import Normal, Bernoulli


data_folder = '../../dat/fake/'
logdir = make_dir('')
fit = pickle.load(open('../fits/fake/EF_EMB_17_06_24_19_08_36/variational9.dat'))

### DATA
left_examples = [data_folder+'train/'+fname for fname in pd.read_csv(data_folder + 'left.txt', header = None)[0].values]
right_examples = [data_folder+'train/'+fname for fname in pd.read_csv(data_folder + 'right.txt', header = None)[0].values]
g_examples = glob.glob(data_folder +'train/g*')


### PARAMETERS

mb = 200
eta = 0.0001
lam = 1.0
n_iter = 50000

### Embeddings
#rho = np.random.randn(L, K)
#alpha = np.random.randn(L, K)
rho = fit['rho']
alpha = fit['alpha']

emb = np.hstack((rho, alpha))
L, K = emb.shape

### Parameters
relevance = tf.nn.sigmoid(tf.Variable(np.random.randn(L).astype('float32')))
print('NOT USING RELEVANCE')
w = tf.Variable(0.001*np.random.randn(K,1).astype('float32'))
#b = tf.Variable(0.0, tf.float32)
b = 0.0

### prior on w
prior = Normal(loc = 0.0, scale = lam)
log_prior = tf.reduce_sum(prior.log_prob(w) + prior.log_prob(b))

### placeholders for data minibatches

def extract_features(text):
     #takes numpy array of text and transforms it into a feature representation
     if len(text) == 0:
         return np.zeros((K))
     return np.mean(emb[text], axis=0)

def next_batch(file_list):
    indices = np.random.permutation(len(file_list))[:mb]
    features = np.zeros((mb, K))
    for i, idx in enumerate(indices):
        features[i] = extract_features(np.load(file_list[idx]))
    return features

### LOGISTIC REGRESSION MODEL
left_ex = tf.placeholder(tf.float32, shape = (mb, K))
right_ex = tf.placeholder(tf.float32, shape = (mb, K))
g_ex = tf.placeholder(tf.float32, shape = (mb, K))

left_eta = b + tf.matmul(left_ex, w)
right_eta = b + tf.matmul(right_ex, w)
g_eta = b + tf.matmul(g_ex, w)

left_y = Bernoulli(logits = left_eta)
right_y = Bernoulli(logits = right_eta)
g_y = Bernoulli(logits = g_eta)

left_bias =  tf.reduce_mean(left_y.log_prob(1.0))
right_bias =  tf.reduce_mean(right_y.log_prob(0.0)) 
neutral =  tf.reduce_mean(g_y.log_prob(0.5))

loss = - (log_prior + 1000.0*(left_bias + right_bias + neutral))

### TRAINING
optimizer = tf.train.AdamOptimizer(eta)
train = optimizer.minimize(loss)
sess = tf.Session()
with sess.as_default():
    tf.global_variables_initializer().run()

saver = tf.train.Saver()
with tf.name_scope('objective'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('neutral', neutral)
    tf.summary.scalar('right_bias', right_bias)
    tf.summary.scalar('left_bias', left_bias)
variable_summaries('weights', w)
variable_summaries('left_eta', left_eta)
variable_summaries('right_eta', right_eta)
variable_summaries('g_eta', g_eta)
summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logdir, sess.graph)



for step in range(n_iter):
    feed_dict = {}
    feed_dict[left_ex] = next_batch(left_examples)
    feed_dict[right_ex] = next_batch(right_examples)
    feed_dict[g_ex] = next_batch(g_examples)

    if step % 10 == 0:
        summary,_ = sess.run([summaries, train], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
    else:
        sess.run([train], feed_dict=feed_dict)
    saver.save(sess, os.path.join(logdir, "model.ckpt"), step)


with sess.as_default():
    dat = {'emb': emb,
           #'b': b.eval(),
           'w': w.eval()}

pickle.dump( dat, open( os.path.join(logdir, "model_params.pkl"), "a+" ) )
