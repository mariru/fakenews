import numpy as np
import os
import pickle
import tensorflow as tf

from scipy.misc import logsumexp
from utils import variable_summaries

from tensorflow.contrib.distributions import Normal, Bernoulli
from tensorflow.contrib.tensorboard.plugins import projector

class emb_model(object):
    def __init__(self, args, d, logdir):
        self.args = args

        self.K = args.K
        self.cs = args.cs
        self.ns = args.ns
        self.n_minibatch = d.n_train
        self.sig = args.sig
        self.unigram = d.unigram
        self.N = d.N
        self.L = d.L
        self.logdir = logdir
        self.train_feed = d.train_feed
        self.eval_states = d.eval_states
        self.n_iter = args.n_iter
        self.n_epochs = d.n_epochs
        self.states = d.states
        self.n_states = d.T
        self.separate = args.separate
        self.completely_separate = args.completely_separate
        self.alpha_trainable = True
        self.rho_trainable = True
        if args.init:
            fname = os.path.join('fits', d.name, args.init)
            if 'rho_constant' in args.init:
                self.rho_trainable = False
                fname = fname.replace('/rho_constant','')
            if 'alpha_constant' in args.init:
                self.alpha_trainable = False
                fname = fname.replace('/alpha_constant','')
            fit = pickle.load(open(fname))
            self.rho_init = fit['rho']
            self.alpha_init = fit['alpha']
        else:
            self.rho_init = (np.random.randn(self.L, self.K)/self.K).astype('float32')
            self.alpha_init = (np.random.randn(self.L, self.K)/self.K).astype('float32')
        if not self.rho_trainable:
            self.alpha_init = (np.random.randn(self.L, self.K)/self.K).astype('float32')
        if not self.alpha_trainable:
            self.rho_init = (np.random.randn(self.L, self.K)/self.K).astype('float32')

        with open(os.path.join(self.logdir,  "log_file.txt"), "a") as text_file:
            text_file.write(str(self.args))
            text_file.write('\n')

    def init_eval_model(self):
        with tf.name_scope('eval_model'):
            self.eval_alpha_state = tf.placeholder(tf.float32)
            self.eval_rho_state = tf.placeholder(tf.float32)
            self.eval_n_test = tf.placeholder(tf.int32)
            eval_n_minibatch = self.eval_n_test - self.cs

            # Data Placeholder
            with tf.name_scope('input'):
                self.eval_ph = tf.placeholder(tf.int32)
                words = self.eval_ph
            

            # Index Masks
            with tf.name_scope('context_mask'):
                p_mask = tf.cast(tf.range(self.cs/2, eval_n_minibatch + self.cs/2), tf.int32)
                rows = tf.cast(tf.tile(tf.expand_dims(tf.range(0, self.cs/2),[0]), [eval_n_minibatch, 1]), tf.int32)
                columns = tf.cast(tf.tile(tf.expand_dims(tf.range(0, eval_n_minibatch), [1]), [1, self.cs/2]), tf.int32)
                ctx_mask = tf.concat([rows+columns, rows+columns +self.cs/2+1], 1)


            with tf.name_scope('natural_param'):
                with tf.name_scope('target_word'):
                    p_idx = tf.gather(words, p_mask)
                    p_rho = tf.squeeze(tf.gather(self.eval_rho_state, p_idx))
                
                # Negative samples
                with tf.name_scope('negative_samples'):
                    self.eval_n_idx = tf.placeholder(tf.int32)
                    n_rho = tf.gather(self.eval_rho_state, self.eval_n_idx)

                with tf.name_scope('context'):
                    ctx_idx = tf.squeeze(tf.gather(words, ctx_mask))
                    ctx_alphas = tf.gather(self.eval_alpha_state, ctx_idx)


                # Natural parameter
                ctx_sum = tf.reduce_sum(ctx_alphas,[1])
                p_eta = tf.expand_dims(tf.reduce_sum(tf.multiply(p_rho, ctx_sum),-1),1)
                n_eta = tf.reduce_sum(tf.multiply(n_rho, tf.tile(tf.expand_dims(ctx_sum,1),[1,self.ns,1])),-1)
            
            # Conditional likelihood
            y_pos = Bernoulli(logits = p_eta)
            y_neg = Bernoulli(logits = n_eta)

            ll_pos = y_pos.log_prob(1.0) 
            ll_neg = tf.reduce_mean(y_neg.log_prob(0.0), axis = 1)
           
            self.eval_ll = tf.nn.moments(ll_pos + ll_neg, axes=[0,1])

    def dump(self, fname):
        raise NotImplementedError()

    def tf_log_likelihood(self, data, t):
        words, neg_words = data.next()
        words = words[:self.n_test[t]]
        neg_words = neg_words[:self.n_test[t] - self.cs,:self.ns]
        with self.sess.as_default():
            if len(self.states)> 0:
                rho_state = self.geo_rho[self.states[t]].eval()
            else:
                rho_state = self.rho.eval()
            if self.completely_separate:
                alpha_state = self.geo_alpha[self.states[t]].eval()
            else:
                alpha_state = self.alpha.eval()
                
        return self.sess.run(self.eval_ll, feed_dict = {self.eval_ph: words, 
                                              self.eval_n_idx: neg_words, 
                                              self.eval_n_test: self.n_test[t], 
                                              self.eval_alpha_state: alpha_state,
                                              self.eval_rho_state: rho_state})

    def evaluate_embeddings(self):
        t_mean = np.array([0.0]*len(self.eval_states))
        t_std = np.array([0.0]*len(self.eval_states))
        v_mean = np.array([0.0]*len(self.eval_states))
        v_std = np.array([0.0]*len(self.eval_states))
        for t, state in enumerate(self.eval_states):
            vm, vs = self.tf_log_likelihood(self.valid_data[state], t)
            tm, ts = self.tf_log_likelihood(self.test_data[state], t)
            t_mean[t] = tm
            t_std[t] = ts
            v_mean[t] = vm
            v_std[t] = vs
            with open(os.path.join(self.logdir,"log_file.txt"), "a") as text_file:
                text_file.write(state)
                text_file.write("\t{:0.5f}\t{:0.5f}".format(v_mean[t], v_std[t]/np.sqrt(self.n_valid[t]))) 
                text_file.write("\t{:0.5f}\t{:0.5f}\n".format(t_mean[t], t_std[t]/np.sqrt(self.n_test[t]))) 
        with open(os.path.join(self.logdir, "..", "log_file.txt"), "a") as text_file:
            text_file.write('\n')
            text_file.write(self.logdir)
            text_file.write('\n')
            text_file.write(str(self.args))
            text_file.write("\nvalid loss: {:0.5f} +- {:0.5f}\n".format(np.average(v_mean, weights = self.n_valid),np.mean(v_std)/np.sqrt(self.n_valid.sum()))) 
            text_file.write("\ntest loss: {:0.5f} +- {:0.5f}\n".format(np.average(t_mean, weights = self.n_test),np.mean(t_std)/np.sqrt(self.n_test.sum()))) 

 
    def initialize_training(self):
        #optimizer = tf.train.GradientDescentOptimizer(self.eta)
        optimizer = tf.train.AdagradOptimizer(0.1)
        self.train = optimizer.minimize(self.loss)
        self.sess = tf.Session()
        with self.sess.as_default():
            tf.global_variables_initializer().run()

        with tf.name_scope('objective'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('priors', self.log_prior)
            tf.summary.scalar('ll_pos', self.ll_pos)
            tf.summary.scalar('ll_neg', self.ll_neg)
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.saver = tf.train.Saver()
        config = projector.ProjectorConfig()
        if not self.completely_separate: 
            alpha = config.embeddings.add()
            alpha.tensor_name = 'model/embeddings/alpha'
            alpha.metadata_path = '../vocab.tsv'
        rho = config.embeddings.add()
        rho.tensor_name = 'model/embeddings/rho'
        rho.metadata_path = '../vocab.tsv'
        for state in self.states:
            rho = config.embeddings.add()
            rho.tensor_name = 'model/embeddings/'+state+'_rho'
            rho.metadata_path = '../vocab.tsv'
            if self.completely_separate:
                alpha = config.embeddings.add()
                alpha.tensor_name = 'model/embeddings/'+state+'_alpha'
                alpha.metadata_path = '../vocab.tsv'
        projector.visualize_embeddings(self.train_writer, config)

    
    def train_embeddings(self):
        for data_pass in range(self.n_iter):
            for step in range(self.n_epochs):
                if step % 10 == 0:
                    print(str(step)+'/'+str(self.n_epochs)+'   iter '+str(data_pass))
                    summary,_ = self.sess.run([self.summaries, self.train], feed_dict=self.train_feed(self.placeholders))
                    self.train_writer.add_summary(summary, data_pass*(self.n_epochs) + step)
                else:
                    self.sess.run([self.train], feed_dict=self.train_feed(self.placeholders))
            self.dump(self.logdir+"/variational"+str(data_pass)+".dat")
            self.saver.save(self.sess, os.path.join(self.logdir, "model.ckpt"), data_pass)



class bern_emb_model(emb_model):
    def __init__(self, args, d, logdir):
        super(bern_emb_model, self).__init__(args, d, logdir)
        self.states = []
        self.n_minibatch = self.n_minibatch.sum()

        with tf.name_scope('model'):
            # Data Placeholder
            with tf.name_scope('input'):
                self.placeholders = tf.placeholder(tf.int32)
                self.words = self.placeholders
            

            # Index Masks
            with tf.name_scope('context_mask'):
                self.p_mask = tf.cast(tf.range(self.cs/2, self.n_minibatch + self.cs/2),tf.int32)
                rows = tf.cast(tf.tile(tf.expand_dims(tf.range(0, self.cs/2),[0]), [self.n_minibatch, 1]),tf.int32)
                columns = tf.cast(tf.tile(tf.expand_dims(tf.range(0, self.n_minibatch), [1]), [1, self.cs/2]),tf.int32)
                self.ctx_mask = tf.concat([rows+columns, rows+columns +self.cs/2+1], 1)

            with tf.name_scope('embeddings'):
                self.rho = tf.Variable(self.rho_init, name='rho', trainable=self.rho_trainable)
                self.alpha = tf.Variable(self.alpha_init, name='alpha', trainable=self.alpha_trainable)

                with tf.name_scope('priors'):
                    prior = Normal(loc = 0.0, scale = self.sig)
                    if self.alpha_trainable:
                        self.log_prior = tf.reduce_sum(prior.log_prob(self.rho) + prior.log_prob(self.alpha))
                    else:
                        self.log_prior = tf.reduce_sum(prior.log_prob(self.rho))

            with tf.name_scope('natural_param'):
                # Taget and Context Indices
                with tf.name_scope('target_word'):
                    self.p_idx = tf.gather(self.words, self.p_mask)
                    self.p_rho = tf.squeeze(tf.gather(self.rho, self.p_idx))
                
                # Negative samples
                with tf.name_scope('negative_samples'):
                    unigram_logits = tf.tile(tf.expand_dims(tf.log(tf.constant(self.unigram)), [0]), [self.n_minibatch, 1])
                    self.n_idx = tf.multinomial(unigram_logits, self.ns)
                    self.n_rho = tf.gather(self.rho, self.n_idx)

                with tf.name_scope('context'):
                    self.ctx_idx = tf.squeeze(tf.gather(self.words, self.ctx_mask))
                    self.ctx_alphas = tf.gather(self.alpha, self.ctx_idx)


                # Natural parameter
                ctx_sum = tf.reduce_sum(self.ctx_alphas,[1])
                self.p_eta = tf.expand_dims(tf.reduce_sum(tf.multiply(self.p_rho, ctx_sum),-1),1)
                self.n_eta = tf.reduce_sum(tf.multiply(self.n_rho, tf.tile(tf.expand_dims(ctx_sum,1),[1,self.ns,1])),-1)
            
            # Conditional likelihood
            self.y_pos = Bernoulli(logits = self.p_eta)
            self.y_neg = Bernoulli(logits = self.n_eta)

            self.ll_pos = tf.reduce_sum(self.y_pos.log_prob(1.0)) 
            self.ll_neg = tf.reduce_sum(self.y_neg.log_prob(0.0))

            self.log_likelihood = self.ll_pos + self.ll_neg
            
            scale = 1.0*self.N/self.n_minibatch
            self.loss = - (self.n_epochs * self.log_likelihood + self.log_prior)

        self.init_eval_model()
            

    def dump(self, fname):
            with self.sess.as_default():
              dat = {'rho':  self.rho.eval(),
                     'alpha':  self.alpha.eval()}
            pickle.dump( dat, open( fname, "a+" ) )


class separate_bern_emb_model(emb_model):
    def __init__(self, args, d, logdir):
        super(separate_bern_emb_model, self).__init__(args, d, logdir)

        with tf.name_scope('model'):
            with tf.name_scope('embeddings'):
                self.alpha = tf.Variable(self.alpha_init, name='alpha', trainable=self.alpha_trainable)

                self.geo_rho = {}
                for t, state in enumerate(d.states):
                    self.geo_rho[state] = tf.Variable(self.rho_init 
                        + 0.001*tf.random_normal([d.L, self.K])/self.K,
                        name = state+'_rho')

                with tf.name_scope('priors'):
                    prior = Normal(loc = 0.0, scale = self.sig)
                    self.log_prior = tf.reduce_sum(prior.log_prob(self.alpha))
                    for state in d.states:
                        self.log_prior += tf.reduce_sum(prior.log_prob(self.geo_rho[state])) 

            with tf.name_scope('likelihood'):
                self.placeholders = {}
                self.y_pos = {}
                self.y_neg = {}
                self.ll_pos = 0.0
                self.ll_neg = 0.0
                for t, state in enumerate(self.states):
                    # Index Masks
                    p_mask = tf.range(self.cs/2,self.n_minibatch[t] + self.cs/2)
                    rows = tf.tile(tf.expand_dims(tf.range(0, self.cs/2),[0]), [self.n_minibatch[t], 1])
                    columns = tf.tile(tf.expand_dims(tf.range(0, self.n_minibatch[t]), [1]), [1, self.cs/2])
                    
                    ctx_mask = tf.concat([rows+columns, rows+columns +self.cs/2+1], 1)

                    # Data Placeholder
                    self.placeholders[state] = tf.placeholder(tf.int32, shape = (self.n_minibatch[t] + self.cs))

                    # Taget and Context Indices
                    p_idx = tf.gather(self.placeholders[state], p_mask)
                    ctx_idx = tf.squeeze(tf.gather(self.placeholders[state], ctx_mask))
                    
                    # Negative samples
                    unigram_logits = tf.tile(tf.expand_dims(tf.log(tf.constant(d.unigram)), [0]), [self.n_minibatch[t], 1])
                    n_idx = tf.multinomial(unigram_logits, self.ns)

                    # Context vectors
                    ctx_alphas = tf.gather(self.alpha, ctx_idx)

                    p_rho = tf.squeeze(tf.gather(self.geo_rho[state], p_idx))
                    n_rho = tf.gather(self.geo_rho[state], n_idx)

                    # Natural parameter
                    ctx_sum = tf.reduce_sum(ctx_alphas,[1])
                    p_eta = tf.expand_dims(tf.reduce_sum(tf.multiply(p_rho, ctx_sum),-1),1)
                    n_eta = tf.reduce_sum(tf.multiply(n_rho, tf.tile(tf.expand_dims(ctx_sum,1),[1,self.ns,1])),-1)
                    
                    # Conditional likelihood
                    self.y_pos[state] = Bernoulli(logits = p_eta)
                    self.y_neg[state] = Bernoulli(logits = n_eta)

                    self.ll_pos += tf.reduce_sum(self.y_pos[state].log_prob(1.0)) 
                    self.ll_neg += tf.reduce_sum(self.y_neg[state].log_prob(0.0))

            self.loss = - (self.n_epochs * (self.ll_pos + self.ll_neg) + self.log_prior)
        self.init_eval_model()

    def dump(self, fname):
            with self.sess.as_default():
                dat = {'alpha':  self.alpha.eval()}
                for state in self.states:
                    dat[state + '_rho'] = self.geo_rho[state].eval()
            pickle.dump( dat, open( fname, "a+" ) )

class completely_separate_bern_emb_model(emb_model):
    def __init__(self, args, d, logdir):
        super(completely_separate_bern_emb_model, self).__init__(args, d, logdir)

        with tf.name_scope('model'):
            with tf.name_scope('embeddings'):
                self.geo_alpha = {}
                self.geo_rho = {}
                for t, state in enumerate(d.states):
                    self.geo_alpha[state] = tf.Variable(self.alpha_init 
                        + 0.001*tf.random_normal([d.L, self.K])/self.K,
                        name = state+'_alpha')
                    self.geo_rho[state] = tf.Variable(self.rho_init 
                        + 0.001*tf.random_normal([d.L, self.K])/self.K,
                        name = state+'_rho')

                with tf.name_scope('priors'):
                    prior = Normal(loc = 0.0, scale = self.sig)
                    self.log_prior = 0.0
                    for state in d.states:
                        self.log_prior += tf.reduce_sum(prior.log_prob(self.geo_rho[state])) 
                        self.log_prior += tf.reduce_sum(prior.log_prob(self.geo_alpha[state])) 

            with tf.name_scope('likelihood'):
                self.placeholders = {}
                self.y_pos = {}
                self.y_neg = {}
                self.ll_pos = 0.0
                self.ll_neg = 0.0
                for t, state in enumerate(self.states):
                    # Index Masks
                    p_mask = tf.range(self.cs/2,self.n_minibatch[t] + self.cs/2)
                    rows = tf.tile(tf.expand_dims(tf.range(0, self.cs/2),[0]), [self.n_minibatch[t], 1])
                    columns = tf.tile(tf.expand_dims(tf.range(0, self.n_minibatch[t]), [1]), [1, self.cs/2])
                    
                    ctx_mask = tf.concat([rows+columns, rows+columns +self.cs/2+1], 1)

                    # Data Placeholder
                    self.placeholders[state] = tf.placeholder(tf.int32, shape = (self.n_minibatch[t] + self.cs))

                    # Taget and Context Indices
                    p_idx = tf.gather(self.placeholders[state], p_mask)
                    ctx_idx = tf.squeeze(tf.gather(self.placeholders[state], ctx_mask))
                    
                    # Negative samples
                    unigram_logits = tf.tile(tf.expand_dims(tf.log(tf.constant(d.unigram)), [0]), [self.n_minibatch[t], 1])
                    n_idx = tf.multinomial(unigram_logits, self.ns)

                    # Context vectors
                    ctx_alphas = tf.gather(self.geo_alpha[state], ctx_idx)

                    p_rho = tf.squeeze(tf.gather(self.geo_rho[state], p_idx))
                    n_rho = tf.gather(self.geo_rho[state], n_idx)

                    # Natural parameter
                    ctx_sum = tf.reduce_sum(ctx_alphas,[1])
                    p_eta = tf.expand_dims(tf.reduce_sum(tf.multiply(p_rho, ctx_sum),-1),1)
                    n_eta = tf.reduce_sum(tf.multiply(n_rho, tf.tile(tf.expand_dims(ctx_sum,1),[1,self.ns,1])),-1)
                    
                    # Conditional likelihood
                    self.y_pos[state] = Bernoulli(logits = p_eta)
                    self.y_neg[state] = Bernoulli(logits = n_eta)

                    self.ll_pos += tf.reduce_sum(self.y_pos[state].log_prob(1.0)) 
                    self.ll_neg += tf.reduce_sum(self.y_neg[state].log_prob(0.0))

            self.loss = - (self.n_epochs * (self.ll_pos + self.ll_neg) + self.log_prior)
        self.init_eval_model()



    def dump(self, fname):
            with self.sess.as_default():
                dat = {}
                for state in self.states:
                    dat[state + '_rho'] = self.geo_rho[state].eval()
                    dat[state + '_alpha'] = self.geo_alpha[state].eval()
            pickle.dump( dat, open( fname, "a+" ) )

def define_model(args, d, logdir):
    if args.separate:
        m = separate_bern_emb_model(args, d, logdir)
    elif args.completely_separate:
        m = completely_separate_bern_emb_model(args, d, logdir)
    elif not (args.separate or args.completely_separate):
        m = bern_emb_model(args, d, logdir)
    else:
        raise NotImplementedError()
    return m

