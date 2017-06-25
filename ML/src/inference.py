from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import os
import pickle
import tensorflow as tf

def define_inference(args, m, dir_name):
    logdir = os.path.join(dir_name, 'tensorboard')
    inference = ed.MAP({}, data = m.data)
    optimizer = tf.train.GradientDescentOptimizer(args.eta)
    inference.initialize(optimizer, logdir=logdir)
    return inference

def train_embeddings(args, dir_name, d, m, inference, sess, saver):
        logdir = os.path.join(dir_name, 'tensorboard')
        saver.save(sess, os.path.join(logdir, "model.ckpt"), 0)
        for i in range(args.n_iter):
            for ii in range(args.n_epochs):
                print(str(ii)+'/'+str(args.n_epochs)+'   iter '+str(i))
                inference.update(d.train_feed(m.placeholders))
            if args.modulated:
                sess.run(m.assign_ops)
            m.dump(dir_name+"/variational"+str(i)+".dat")
            saver.save(sess, os.path.join(logdir, "model.ckpt"), i)
