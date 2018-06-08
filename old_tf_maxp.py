import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.python.ops import init_ops
from tensorflow.contrib import rnn
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops import math_ops

import data.yahoo_reader3 as reader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
plt_x = []
plt_y11 = []
plt_y12 = []
plt_y13 = []
plt_y21 = []
plt_y22 = []
plt_y23 = []
plt_y31 = []
plt_y32 = []
plt_y33 = []


def embedding_initializer(embedding_init):
    print('Warning -- This is embedding_initializer function')
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      if len(shape) == 2:
        return tf.constant(embedding_init, dtype=tf.float32)
      else:
        raise ValueError("embedding_initializer::shape is wrong! %s", len(shape))
    return _initializer


class Config(object):
    """config."""
    def __init__(self):
        self.init_scale = 0.04
        self.max_grad_norm = 5.
        self.save_model = True  # save model or not
        self.best_ans = 0.0  # for save model and early stopping
        self.current_best_ans = 0.0
        self.restore = False  # restore model or not
        self.train = True
        self.initializer = None #"orthogonal" or None
        self.clip = None # "global"
        self.print_debug_info = False
        self.save_name = "yahooQA_maxp_1002"
        self.restore_name = "yahooQA_maxp_1002"
        self.save_path = "models/"+self.save_name+".tf.variables"  # model path to restore
        self.restore_path = "models/"+self.restore_name+".tf.variables"
        self.cuda = '3'

        # Parameters
        self.margin = 0.05
        self.weight_decay = None
        self.keep_prob = 1.0
        self.keep_output_prob = 1.0
        self.keep_input_prob = 0.7
        self.learning_rate = 2e-5
        self.lr_init = self.learning_rate
        self.lr_decay_epoch_begin = 1  # lr decay start
        self.lr_decay_epoch_end = 50 # lr decay stop
        self.lr_decay_epoch_step = None # lr decaysave_path every step, set to None for non decay
        self.lr_decay_rate = 0.8
        self.embedding_tuning = True
        self.dev_accuracy = True #using dev accuracy to chose parameter

        self.training_eps = 50 # training epoches
        self.batch_size = 512 #
        self.evaluate_batch_size = 1 # re calculate the embedding of A for each Q
        self.display_step = 10 # display training info every 10 mini-batches.
        self.valid_epoch = 1 # perform validation every valid_epoch

        # Network Parameters
        self.n_input = 300 # word embedding
        self.n_steps = 50 # timesteps, sequence max len, over that will be discarded.
        self.n_hidden = 50 # hidden layer num of features
        self.n_output = self.n_hidden * 2
        self.vocab_size = 0
        self.answer_num = 0

    def __str__(self):
        attrs = vars(self)
        print_attrs = '\n '.join("%s: %s" % item for item in attrs.items())
        return print_attrs


class RNNs_Model(object):
    def __init__(self, config, isTrain, embedding_init = None):
        # tf Graph input
        bn = None
        self._Q = tf.placeholder(tf.int32, [bn, config.n_steps])
        self._A = tf.placeholder(tf.int32, [bn, config.n_steps])
        self._Neg = tf.placeholder(tf.int32, [bn, config.n_steps])
        self._Qseq = tf.placeholder(tf.int32, [bn])
        self._Aseq = tf.placeholder(tf.int32, [bn])
        self._Negseq = tf.placeholder(tf.int32, [bn])
        self._Qmask = tf.placeholder("float", [bn, config.n_steps, 1])
        self._Amask = tf.placeholder("float", [bn, config.n_steps, 1])
        self._Negmask = tf.placeholder("float", [bn, config.n_steps, 1])

        current_batch_size = tf.shape(self._Q)[0]
        # Define weights
        weights = {
            'h0_f': tf.get_variable("h0_f", [1, config.n_hidden]),
            'h0_b': tf.get_variable("h0_b", [1, config.n_hidden]),
            'c0_f': tf.get_variable("c0_f", [1, config.n_hidden]),
            'c0_b': tf.get_variable("c0_b", [1, config.n_hidden]),
        }
        biases = {

        }

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", shape=[config.vocab_size, config.n_input], dtype=tf.float32,
                                        initializer=embedding_initializer(embedding_init), trainable=config.embedding_tuning)
            input_Q = tf.nn.embedding_lookup(embedding, self._Q)
            input_A = tf.nn.embedding_lookup(embedding, self._A)
            input_Neg = tf.nn.embedding_lookup(embedding, self._Neg)

        if isTrain and config.keep_input_prob < 1:
            input_Q = tf.nn.dropout(input_Q, config.keep_input_prob)
            input_A = tf.nn.dropout(input_A, config.keep_input_prob)
            input_Neg = tf.nn.dropout(input_Neg, config.keep_input_prob)

        def BRNN(_X, _Xseq, _MASK, _weights, _biases):

            cell_f = rnn.LSTMCell(config.n_hidden)
            cell_b = rnn.LSTMCell(config.n_hidden)

            if isTrain and config.keep_prob < 1:
                cell_f = rnn.DropoutWrapper(cell_f, output_keep_prob=config.keep_prob)
                cell_b = rnn.DropoutWrapper(cell_b, output_keep_prob=config.keep_prob)

            _h0_f = tf.tile(_weights["h0_f"], [current_batch_size, 1])
            _h0_b = tf.tile(_weights["h0_b"], [current_batch_size, 1])
            _c0_f = tf.tile(_weights["c0_f"], [current_batch_size, 1])
            _c0_b = tf.tile(_weights["c0_b"], [current_batch_size, 1])

            _initial_state_f = rnn.LSTMStateTuple(_c0_f, _h0_f)
            _initial_state_b = rnn.LSTMStateTuple(_c0_b, _h0_b)

            (outputs_f, outputs_b), states = tf.nn.bidirectional_dynamic_rnn(cell_f, cell_b, _X,
                                                  sequence_length = _Xseq,
                                                  initial_state_fw =_initial_state_f,
                                                  initial_state_bw =_initial_state_b)

            outputs = tf.concat(values=[outputs_f, outputs_b], axis=2)
            if isTrain and config.keep_output_prob < 1:
                outputs = tf.nn.dropout(outputs, config.keep_output_prob)

            return outputs

        def pooling(tensor, mask):
            # avg pooling
            #matrix = tf.reduce_sum(tensor, reduction_indices=1)
            #return matrix/tf.reduce_sum(mask, reduction_indices=[1,2])[:,None]

            # max pooling
            t = tensor - tf.constant(9999, dtype=tf.float32) * tf.ones_like(tensor, dtype=tf.float32) * (tf.ones_like(mask) - mask)
            matrix = tf.reduce_max(t, reduction_indices=1)

            return matrix

        def similar(Q, A):
            # Q,A(batchsize, hidden*2)
            # sim (batchsize, )

            # sim = tf.reduce_sum(Q * A, reduction_indices=1)
            sim = tf.reduce_sum(tf.nn.l2_normalize(Q, dim=1) * tf.nn.l2_normalize(A, dim=1), reduction_indices=1)
            return sim

        with tf.variable_scope("BRNNs", reuse=None):
            outputs = BRNN(input_Q, self._Qseq, self._Qmask, weights, biases)
            self._Qoutput = pooling(outputs, self._Qmask)

        with tf.variable_scope("BRNNs", reuse=True):
            outputs = BRNN(input_A, self._Aseq, self._Amask, weights, biases)
            self._Aoutput = pooling(outputs, self._Amask)

        with tf.variable_scope("BRNNs", reuse=True):
            outputs = BRNN(input_Neg, self._Negseq, self._Negmask, weights, biases)
            self._Negoutput = pooling(outputs, self._Negmask)


        if not isTrain:
            return

        Sim_QA = similar(self._Qoutput, self._Aoutput)
        Sim_QNeg = similar(self._Qoutput, self._Negoutput)

        # cost is (batch size, )
        cost = tf.maximum(tf.zeros_like(Sim_QA), config.margin - Sim_QA + Sim_QNeg)
        self._cost = tf.reduce_mean(cost)

        tvars = tf.trainable_variables()
        for v in tvars:
            print v.name

        self._vnames = [v.name for v in tvars]
        self._vars = tvars
        self._lr = tf.Variable(0.0, trainable=False)

        if config.weight_decay is not None:
            for v in tvars:
                if "weights" in v.name or "W" in v.name:
                    weights_norm = tf.reduce_sum(tf.sqrt(tf.nn.l2_loss(v)))
                    self._cost += config.weight_decay * weights_norm
                    print "weigth decay for ", v.name

        # use clip by norm
        if config.clip == "global":
            grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), config.max_grad_norm)
        else:
            grads =[tf.clip_by_norm(grad, config.max_grad_norm) for grad in tf.gradients(self._cost, tvars)]

        optimizer_tmp = tf.train.RMSPropOptimizer(self._lr, decay= 0.5, momentum=0.9)
        #optimizer_tmp = tf.train.GradientDescentOptimizer(self._lr)
        self._optimizer = optimizer_tmp.apply_gradients(zip(grads, tvars))


        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def alphas(self):
        return self._S

    @property
    def Output(self):
        return self._Qoutput

    @property
    def X(self):
        return self._Q

    @property
    def Seq(self):
        return self._Qseq

    @property
    def Mask(self):
        return self._Qmask

    @property
    def Aoutput(self):
        return self._Aoutput

    @property
    def cost(self):
        return self._cost

    @property
    def vars(self):
        return self._vars

    @property
    def vnames(self):
        return self._vnames

    @property
    def lr(self):
        return self._lr

    @property
    def optimizer(self):
        return self._optimizer

def draw_figure(config):
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)

    plt.sca(ax1)
    line1, = plt.plot(plt_x, plt_y11, label="train", color ="green")
    line2, = plt.plot(plt_x, plt_y12, label="valid", color="red")
    line3, = plt.plot(plt_x, plt_y13, label="test", color="black")
    plt.xlabel('epoches')
    plt.ylabel('ACC')
    plt.legend([line1, line2, line3], ["train", "valid", "test"], loc="lower right")
    plt.grid(True, linestyle='-', linewidth=1)

    plt.sca(ax2)
    line4, = plt.plot(plt_x, plt_y21, label="train", color ="green")
    line5, = plt.plot(plt_x, plt_y22, label="valid", color="red")
    line6, = plt.plot(plt_x, plt_y23, label="test", color="black")
    plt.xlabel('epoches')
    plt.ylabel('MAP')
    plt.legend([line4, line5, line6], ["train", "valid", "test"], loc="lower right")
    plt.grid(True, linestyle='-', linewidth=1)

    plt.sca(ax3)
    line7, = plt.plot(plt_x, plt_y31, label="train", color ="green")
    line8, = plt.plot(plt_x, plt_y32, label="valid", color="red")
    line9, = plt.plot(plt_x, plt_y33, label="test", color="black")
    plt.xlabel('epoches')
    plt.ylabel('MRR')
    plt.legend([line7, line8, line9], ["train", "valid", "test"], loc="lower right")
    plt.grid(True, linestyle='-', linewidth=1)

    savefig(config.save_name+".jpg")

def similar(q, a):
    # return np.inner(q, a)
    q_normalized = q / np.linalg.norm(q, ord=2, axis=0)
    a_normalized = a / np.linalg.norm(a, ord=2, axis=1)[:,None]
    return  np.inner(q_normalized, a_normalized)

def similar_fast(q, a):
    q_normalized = q / np.linalg.norm(q, ord=2, axis=1)[:,None]
    a_normalized = a / np.linalg.norm(a, ord=2, axis=1)[:,None]
    return  np.sum(q_normalized * a_normalized, axis=1)

def _acc(sim, groundTruth):
    right = 0
    y = np.argmax(sim)

    if y < groundTruth:
        return 1
    else:
        return 0

def _map_mrr(sim, groundTruth):
    ind = np.argsort(sim)
    rank = []
    for i in range(len(ind)):
        for j in range(len(ind)):
            if ind[j] == i:
                rank.append(j)
    rank = len(ind) - np.array(rank)

    map = 0.0
    ranksort = np.sort(rank[:groundTruth])
    for j in range(groundTruth):
        map += float(j+1)/float(ranksort[j])
    map /= groundTruth

    #MRR
    mrr = float(1)/float(ranksort[0])

    return map, mrr

def evaluate_acc(sess, config, model, data_iter, check_alpha=False, ds=None):
    st = time.time()
    acc_sum = 0.0
    map_sum = 0.0
    mrr_sum = 0.0
    dataN = 0
    acc_s = {}
    acc_n = {}
    for mb_q, mb_qseq, mb_qmask, mb_a, mb_aseq, mb_amask, gd, judge, c in data_iter:
        Q, A = sess.run([model.Output, model.Aoutput], feed_dict={
                                model.X: mb_q, model.Seq: mb_qseq, model.Mask: mb_qmask,
                                model._A: mb_a, model._Aseq: mb_aseq, model._Amask: mb_amask})

        sims = similar_fast(Q, A)
        idx = 0
        for i in range(len(judge)):
            sim = sims[idx: idx+judge[i]]
            idx = idx + judge[i]
            mb_gd = gd[i]
            acc = _acc(sim, mb_gd)
            acc_sum += acc
            if acc_s.get(c[i]) is None:
                acc_s[c[i]] = acc
                acc_n[c[i]] = 1
            else:
                acc_s[c[i]] += acc
                acc_n[c[i]] += 1

            map, mrr = _map_mrr(sim, mb_gd)
            map_sum += map
            mrr_sum += mrr
            dataN += 1
    for k, v in acc_s.items():
        acc_s[k] = float(acc_s[k])/acc_n[k]
    T =  "eval cost time: "+str(time.time() - st)+"seconds"
    print T

    return acc_sum / dataN, map_sum / dataN, mrr_sum / dataN, acc_s, acc_n

def main(_):
    main_start_time = time.strftime("%d %b %Y %H:%M:%S", time.localtime())

    r = np.random.randint(0, 100)
    r = float(r) / 1000 + 0.2
    gpuconfig = tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=r),
        #device_count = {'GPU': 11}
    )

    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda


    # Launch the graph
    with tf.Graph().as_default(), tf.Session(config = gpuconfig) as sess:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        ####### loading data ########
        ds = reader.yahoo_reader(deflen = config.n_steps, n_input=config.n_input)
        train_data, valid_data, test_data, embedding_init, train_for_test_data = ds.loading()
        config.vocab_size = len(ds.vocab)
        #############################

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtrain = RNNs_Model(config, isTrain = True, embedding_init=embedding_init)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = RNNs_Model(config, isTrain = False)

        print "Build graph done!"
        if config.restore:
            print "session restoring..."
            saver = tf.train.Saver()
            saver.restore(sess, config.restore_path)
            print "session restored!"
        else:
            tf.initialize_all_variables().run()

        if config.restore:
            shelllog = open('shelllog/'+str(config.save_name)+'.log','a+')
            shelllog.write("restore... \n config:      "+str(config)+"\n\n\n")
        else:
            shelllog = open('shelllog/'+str(config.save_name)+'.log','w+')
            shelllog.write("program config:      "+str(config)+"\n\n\n")

        checklog = open('checklog/'+str(config.save_name)+'.checklog','w+')

        eps = step = 0
        while eps < config.training_eps and config.train:
            mbs = 1
            mtrain.assign_lr(sess, config.learning_rate)
            batch_time = time.time()
            train_epoch_iter = ds.epoch_iterator(train_data, config.batch_size)
            eps_start_time = time.time()
            for mb_q, mb_qseq, mb_qmask, mb_a, mb_aseq, mb_amask, mb_neg, mb_negseq, mb_negmask in train_epoch_iter:
                step = step + 1

                if step % config.display_step != 0:
                    # Fit training using batch data
                    sess.run(mtrain.optimizer, feed_dict={mtrain._Q: mb_q, mtrain._Qseq: mb_qseq, mtrain._Qmask: mb_qmask,
                                                      mtrain._A: mb_a, mtrain._Aseq: mb_aseq, mtrain._Amask: mb_amask,
                                                      mtrain._Neg: mb_neg, mtrain._Negseq: mb_negseq, mtrain._Negmask: mb_negmask,})
                else:
                    # Calculate batch accuracy and loss
                    [loss, vars, _] = sess.run([mtrain.cost, mtrain.vars, mtrain.optimizer],
                                            feed_dict={mtrain._Q: mb_q, mtrain._Qseq: mb_qseq, mtrain._Qmask: mb_qmask,
                                                      mtrain._A: mb_a, mtrain._Aseq: mb_aseq, mtrain._Amask: mb_amask,
                                                      mtrain._Neg: mb_neg, mtrain._Negseq: mb_negseq, mtrain._Negmask: mb_negmask,})

                    infos = "EP "+str(eps+1)+", MB "+str(mbs)+", lr "+str(config.learning_rate)+", MB Loss= "+ \
                            "{:.4f}".format(loss) +", Time:{:.4f}".format(time.time() - batch_time)
                    batch_time = time.time()
                    print infos
                    shelllog.write(infos+"\n")

                    if config.print_debug_info:
                        param_norms = ""
                        for i in range(len(mtrain.vnames)):
                            param_norms += "{:.4f}".format(np.linalg.norm(vars[i]))+"\t"+mtrain.vnames[i]+"\n"
                        param_norms +="\n"

                        print param_norms
                        shelllog.write(param_norms)

                mbs += 1
            print "every single epoch cost time:", time.time() - eps_start_time
            if (eps+1) % config.valid_epoch == 0:

                valid_epoch_iter = ds.epoch_test_iterator(valid_data)
                acc_valid, map_valid, mrr_valid, accs_valid, _ = evaluate_acc(sess, config, mtest, valid_epoch_iter)

                test_epoch_iter = ds.epoch_test_iterator(test_data)
                acc_test, map_test, mrr_test, accs_test, accn = evaluate_acc(sess, config, mtest, test_epoch_iter)

                #train_epoch_iter = ds.epoch_test_iterator(train_for_test_data)
                #acc_train, map_train, mrr_train, accs_train, _ = evaluate_acc(sess, config, mtest, train_epoch_iter)
                acc_train, map_train, mrr_train, accs_train = acc_valid, map_valid, mrr_valid, accs_valid

                plt_x.append(eps)
                plt_y11.append(acc_train)
                plt_y12.append(acc_valid)
                plt_y13.append(acc_test)

                plt_y21.append(map_train)
                plt_y22.append(map_valid)
                plt_y23.append(map_test)

                plt_y31.append(mrr_train)
                plt_y32.append(mrr_valid)
                plt_y33.append(mrr_test)

                print "Valid Acc: ", "{:.4f}".format(acc_valid), "Test Acc: ", "{:.4f}".format(acc_test),"Train Acc: ", "{:.4f}".format(acc_train)
                print "Valid map: ", "{:.4f}".format(map_valid), "Test map: ", "{:.4f}".format(map_test),"Train map: ", "{:.4f}".format(map_train)
                print "Valid mrr: ", "{:.4f}".format(mrr_valid), "Test mrr: ", "{:.4f}".format(mrr_test),"Train mrr: ", "{:.4f}".format(mrr_train)

                for k, v in accs_test.items():
                    print k, "Test Acc:{:.4f}".format(accs_test[k])
                    shelllog.write(k + "Test Acc:{:.4f}".format(accs_test[k])+"   N:"+str(accn[k])+"\n")

                shelllog.write("Validation Accuracy: "+"{:.4f}".format(acc_valid)+
                               " Test Accuracy: "+"{:.4f}".format(acc_test)+"\n")
                shelllog.write("Train Accuracy: "+ "{:.4f}".format(acc_train)+"\n\n")

                shelllog.write("Validation Map: "+"{:.4f}".format(map_valid)+
                               " Test Map: "+"{:.4f}".format(map_test)+"\n")
                shelllog.write("Train Map: "+ "{:.4f}".format(map_train)+"\n\n")

                shelllog.write("Validation mrr: "+"{:.4f}".format(mrr_valid)+
                               " Test mrr: "+"{:.4f}".format(mrr_test)+"\n")
                shelllog.write("Train mrr: "+ "{:.4f}".format(mrr_train)+"\n\n")
                print "\n"

                draw_figure(config)

                if config.dev_accuracy:
                    chose_valid = acc_valid
                    chose_test = acc_test
                else:
                    chose_valid = map_valid
                    chose_test = map_test

                if config.save_model and chose_valid >= config.best_ans:
                    config.best_ans = chose_valid
                    config.current_best_ans = chose_test
                    saver = tf.train.Saver()
                    config.save_path = saver.save(sess, config.save_path)

                print "Current Best Test:", "{:.4f}".format(config.current_best_ans)
                shelllog.write("Current Best Test:"+"{:.4f}".format(config.current_best_ans)+"\n")

            eps += 1

            if config.lr_decay_epoch_step is not None:
                if config.lr_decay_epoch_begin <= eps and eps <= config.lr_decay_epoch_end:
                    if (eps - config.lr_decay_epoch_begin) % config.lr_decay_epoch_step == 0:
                        config.learning_rate *= config.lr_decay_rate
                        print "learning rate decay!  learning rate: ", config.learning_rate


        print "Optimization Finished!"


        print "session recovering... loading the best parameters..."
        saver = tf.train.Saver()
        saver.restore(sess, config.save_path)
        print "session recovered!"

        valid_epoch_iter = ds.epoch_test_iterator(valid_data)
        acc_valid, map_valid, mrr_valid, accs_valid, _ = evaluate_acc(sess, config, mtest, valid_epoch_iter)

        test_epoch_iter = ds.epoch_test_iterator(test_data)
        acc_test, map_test, mrr_test, accs_test, accn = evaluate_acc(sess, config, mtest, test_epoch_iter, True, ds)

        train_epoch_iter = ds.epoch_test_iterator(train_for_test_data)
        acc_train, map_train, mrr_train, accs_train, _ = evaluate_acc(sess, config, mtest, train_epoch_iter)

        print "Valid Acc: ", "{:.4f}".format(acc_valid), "Test Acc: ", "{:.4f}".format(acc_test),"Train Acc: ", "{:.4f}".format(acc_train)
        print "Valid map: ", "{:.4f}".format(map_valid), "Test map: ", "{:.4f}".format(map_test),"Train map: ", "{:.4f}".format(map_train)
        print "Valid mrr: ", "{:.4f}".format(mrr_valid), "Test mrr: ", "{:.4f}".format(mrr_test),"Train mrr: ", "{:.4f}".format(mrr_train)

        for k, v in accs_test.items():
            print k, "Test Acc:{:.4f}".format(accs_test[k])
            shelllog.write(k + "Test Acc:{:.4f}".format(accs_test[k])+"   N:"+str(accn[k])+"\n")

        shelllog.write("Validation Accuracy: "+"{:.4f}".format(acc_valid)+
                               " Test Accuracy: "+"{:.4f}".format(acc_test)+"\n")
        shelllog.write("Train Accuracy: "+ "{:.4f}".format(acc_train)+"\n\n")

        shelllog.write("Validation Map: "+"{:.4f}".format(map_valid)+
                               " Test Map: "+"{:.4f}".format(map_test)+"\n")
        shelllog.write("Train Map: "+ "{:.4f}".format(map_train)+"\n\n")

        shelllog.write("Validation mrr: "+"{:.4f}".format(mrr_valid)+
                               " Test mrr: "+"{:.4f}".format(mrr_test)+"\n")
        shelllog.write("Train mrr: "+ "{:.4f}".format(mrr_train)+"\n\n")

        runninglog = open('lstm_log.txt','a+')
        runninglog.write("program start time:  "+str(main_start_time)+"\n")
        print "program start time: "+str(main_start_time)
        print "program end time:   "+str(time.strftime("%d %b %Y %H:%M:%S", time.localtime()))
        runninglog.write("program finish time: "+str(time.strftime("%d %b %Y %H:%M:%S", time.localtime()))+"\n")
        runninglog.write("program config:      "+str(config)+"\n\n\n")
        runninglog.write("\n")
        runninglog.close()
        shelllog.close()
        checklog.close()

if __name__ == "__main__":
    try:
        tf.app.run()
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
    except KeyboardInterrupt:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'

