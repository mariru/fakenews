import glob
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from utils import *
from tensorflow.contrib.distributions import Normal, Bernoulli


data_folder = '../../dat/fake/'
logdir = make_dir('')
fit = pickle.load(open('../fits/fake/EF_EMB_17_06_24_19_08_36/variational6.dat'))

### DATA
left_examples = [data_folder+'train/'+fname for fname in pd.read_csv(data_folder + 'left.txt', header = None)[0].values]
right_examples = [data_folder+'train/'+fname for fname in pd.read_csv(data_folder + 'right.txt', header = None)[0].values]
g_examples = glob.glob(data_folder +'train/g*')


### PARAMETERS

K = 20
L = 15000
mb = 500
eta = 0.1
lam = 1.0
n_iter = 5000

### Embeddings
#rho = np.random.randn(L, K)
#alpha = np.random.randn(L, K)
rho = fit['rho']
alpha = fit['alpha']

emb = np.hstack((rho, alpha))
K = emb.shape[1]

### Parameters
relevance = tf.nn.sigmoid(tf.Variable(np.random.randn(L).astype('float32')))
print('NOT USING RELEVANCE')
w = tf.Variable(np.random.randn(K,1).astype('float32'))

### prior on w
prior = Normal(loc = 0.0, scale = lam)
log_prior = tf.reduce_sum(prior.log_prob(w))

### placeholders for data minibatches

def extract_features(text):
     #takes numpy array of text and transforms it into a feature representation
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

left_eta = tf.matmul(left_ex, w)
right_eta = tf.matmul(right_ex, w)
g_eta = tf.matmul(g_ex, w)

left_y = Bernoulli(logits = left_eta)
right_y = Bernoulli(logits = right_eta)
g_y = Bernoulli(logits = g_eta)

left_bias =  tf.reduce_sum(left_y.log_prob(1.0))
right_bias =  tf.reduce_sum(right_y.log_prob(0.0)) 
neutral =  tf.reduce_sum(g_y.log_prob(0.5))

loss = log_prior + left_bias + right_bias + neutral

### TRAINING
optimizer = tf.train.AdagradOptimizer(eta)
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
variable_summaries('weights',w)
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
  dat = {'rho':  rho.eval(),
         'alpha':  alpha.eval(),
         'w': w.eval()}
pickle.dump( dat, open( os.path.join(logdir, "model_params.pkl"), "a+" ) )
