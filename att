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
            'c0_f': tf.get_variable("c0_f", [1, config.n_hidden]),
            'h0_b': tf.get_variable("h0_b", [1, config.n_hidden]),
            'c0_b': tf.get_variable("c0_b", [1, config.n_hidden]),
        }
        biases = {

        }

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", shape=[config.vocab_size+1, config.n_input], dtype=tf.float32,
                                        initializer=embedding_initializer(embedding_init))
            input_Q = tf.nn.embedding_lookup(embedding, self._Q)
            input_A = tf.nn.embedding_lookup(embedding, self._A)
            input_Neg = tf.nn.embedding_lookup(embedding, self._Neg)

        if isTrain and config.keep_input_prob < 1:
            input_Q = tf.nn.dropout(input_Q, config.keep_input_prob)
            input_A = tf.nn.dropout(input_A, config.keep_input_prob)
            input_Neg = tf.nn.dropout(input_Neg, config.keep_input_prob)

        def BRNN(_X, _Xseq, _MASK, _weights, _biases):

            lstm_cell_f = rnn.LSTMCell(config.n_hidden, forget_bias=1.0)
            lstm_cell_b = rnn.LSTMCell(config.n_hidden, forget_bias=1.0)

            if isTrain and config.keep_prob < 1:
                lstm_cell_f = rnn.DropoutWrapper(lstm_cell_f, output_keep_prob=config.keep_prob)
                lstm_cell_b = rnn.DropoutWrapper(lstm_cell_b, output_keep_prob=config.keep_prob)

            _c0_f = tf.tile(_weights["c0_f"], [current_batch_size, 1])
            _h0_f = tf.tile(_weights["h0_f"], [current_batch_size, 1])
            _c0_b = tf.tile(_weights["c0_b"], [current_batch_size, 1])
            _h0_b = tf.tile(_weights["h0_b"], [current_batch_size, 1])

            _initial_state_f = rnn.LSTMStateTuple(_c0_f, _h0_f)
            _initial_state_b = rnn.LSTMStateTuple(_c0_b, _h0_b)

            (outputs_f, outputs_b), states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_f, lstm_cell_b, _X,
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

        def attention(A, mask, q):
            H = A
            M = tf.reduce_sum(H * q[:,None,:], axis=2)[:,:,None]
            S_tmp = math_ops.exp(M)
            S_tmp = S_tmp * mask
            S = S_tmp / tf.reduce_sum(S_tmp, axis=1)[:,None,:]

            H_hat = H * S
            return tf.reduce_sum(H_hat, axis=1), S



        with tf.variable_scope("BRNNs", reuse=None):
            outputs = BRNN(input_Q, self._Qseq, self._Qmask, weights, biases)
            self._Qoutput = pooling(outputs, self._Qmask)

        with tf.variable_scope("BRNNs", reuse=True):
            outputs = BRNN(input_A, self._Aseq, self._Amask, weights, biases)
            self._Aoutput, self._S = attention(outputs, self._Amask, self._Qoutput)

        with tf.variable_scope("BRNNs", reuse=True):
            outputs = BRNN(input_Neg, self._Negseq, self._Negmask, weights, biases)
            self._Negoutput, S = attention(outputs, self._Negmask, self._Qoutput)


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
