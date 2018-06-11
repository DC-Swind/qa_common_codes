
def DictToList(d):
    l = []
    for i in range(len(d)):
        l.append(d.get(i))
    return np.array(l)

def draw_figure(config):

    line1, = plt.plot(plt_x, plt_y, label="train", color ="green")
    line2, = plt.plot(plt_x, plt_y2, label="valid", color="red")
    line3, = plt.plot(plt_x, plt_y3, label="test1", color="black")
    line4, = plt.plot(plt_x, plt_y4, label="test2", color="blue")

    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.legend([line1, line2, line3, line4], ["train", "valid", "test1", "test2"], loc="lower right")
    plt.grid(True, linestyle='-', linewidth=1)
    savefig(config.save_name+".jpg")

def ans_forward(sess, config, model, ans_iter):
    embeddings = np.zeros((config.answer_num, config.n_output))
    index = 0
    for mb_a, mb_seqa, mb_maska in ans_iter:
        # embeddings shape is (batchsize, dim output)
        mb_embeddings = sess.run(model.Output, feed_dict={model.X: mb_a, model.Seq: mb_seqa, model.Mask: mb_maska})
        embeddings[index:index + mb_a.shape[0],:] = mb_embeddings
        index += mb_a.shape[0]
    return embeddings

def similar(q, a):
    # return np.inner(q, a)
    q_normalized = q / np.linalg.norm(q, ord=2, axis=0)
    a_normalized = a / np.linalg.norm(a, ord=2, axis=1)[:,None]
    return  np.inner(q_normalized, a_normalized)

def _acc(Qembed, GroundTruths, Candidates, Aembed, config):
    right = 0
    # q is (dim, )
    q = Qembed
    # a is (Cds, dim)
    a = Aembed

    sim = similar(q, a)
    y = np.argmax(sim)

    if Candidates[y] in GroundTruths:
        return 1, y, y
    else:
        for i in range(sim.shape[0]):
            if Candidates[i] in GroundTruths:
                return 0, y, i
        return 0, y, -1

def evaluate_acc(sess, config, model, data_iter, answers, ds, shelllog = None, check=False):
    st = time.time()
    acc_sum = 0.0
    dataN = 0
    for mb_q, mb_qseq, mb_qmask, mb_gt, mb_cd in data_iter:
        cds = answers[mb_cd[0]]
        ans_iter = ds.test_Answer_iterator(len(cds), cds)
        mb_q = np.tile(mb_q, [len(cds), 1])
        mb_qseq = np.tile(mb_qseq, [len(cds)])
        mb_qmask = np.tile(mb_qmask, [len(cds), 1, 1])


        for mb_a, mb_aseq, mb_amask in ans_iter:
            # Q, A is (cds, dim)

            Q, A, alphas = sess.run([model._Qoutput, model._Aoutput, model.alphas], feed_dict={
                                        model._Q: mb_q, model._Qseq: mb_qseq, model._Qmask: mb_qmask,
                                        model._A: mb_a, model._Aseq: mb_aseq, model._Amask: mb_amask})
            acc, y, t = _acc(Q[0,:], mb_gt[0], mb_cd[0], A, config)
            acc_sum += acc
            if check:
                if y == t:
                    outputInfo(mb_q[0,:], mb_a[y,:], alphas[y,:], mb_a[t,:], alphas[t,:], 1, config, ds)
                elif t == -1:
                    print "error!"
                else:
                    outputInfo(mb_q[0,:], mb_a[y,:], alphas[y,:], mb_a[t,:], alphas[t,:], 0, config, ds)
            dataN += 1


    T =  "eval cost time: "+str(time.time() - st)+"seconds\n"
    print T
    if shelllog is not None:
        shelllog.write(T)

    f = open("checklog/"+config.save_name+".log", "a+")
    f.write("\n\n\n\n\n")
    f.close()
    return acc_sum / dataN

def outputInfo(q_Indexes, a_Indexes, a_alpha, t_Indexes, t_alpha, right, config, ds):
    f = open("checklog/"+config.save_name+".log", "a+")
    q = ds.convert_index_to_word(q_Indexes)
    a = ds.convert_index_to_word(a_Indexes)
    t = ds.convert_index_to_word(t_Indexes)
    f.write(str(right)+"\n")
    for w in q:
        f.write(w+" ")
    f.write("\n")
    for w, alpha in zip(a, a_alpha):
        f.write(w+"~"+"{:.6f}".format(alpha[0])+" ")
    f.write("\n")
    for w, alpha in zip(t, t_alpha):
        f.write(w+"~"+"{:.6f}".format(alpha[0])+" ")
    f.write("\n\n")
    f.close()

def main(_):
    main_start_time = time.strftime("%d %b %Y %H:%M:%S", time.localtime())

    gpuconfig = tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
        #device_count = {'GPU': 11}
    )

    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda

    ####### loading data ########
    ds = reader.Insurance_reader()
    train_data, valid_data, test1_data, test2_data, train_data_for_test = ds.loading()
    embedding_init = ds.embeddingInit()
    config.vocab_size = len(ds.Vocab)
    config.answer_num = len(ds.Answers)
    #############################

    # Launch the graph
    with tf.Graph().as_default(), tf.Session(config = gpuconfig) as sess:

        if config.initializer == "orthogonal":
            initializer = ltf.init_ops.orthogonal_initializer(scale = config.init_scale)
        else:
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtrain = RNNs_Model(config, isTrain = True, embedding_init=embedding_init)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = RNNs_Model(config, isTrain = False)

        print "Build graph done!"
        if config.restore:
            print "session restoring..."
            saver = tf.train.Saver()
            saver.restore(sess, config.restore_path)
            saver.save(sess, config.save_path)
            print "session restored!"
        else:
            tf.initialize_all_variables().run()



        if config.restore:
            shelllog = open('shelllog/'+str(config.save_name)+'.log','a+')
            shelllog.write("restore... \n config:      "+str(config)+"\n\n\n")
        else:
            shelllog = open('shelllog/'+str(config.save_name)+'.log','w+')
            shelllog.write("program config:      "+str(config)+"\n\n\n")

        checklog = open('checklog/'+str(config.save_name)+'.log','w+')

        eps = step = 0
        while eps < config.training_eps and config.train:
            mbs = 1
            mtrain.assign_lr(sess, config.learning_rate)

            train_epoch_iter = ds.epoch_iterator(train_data, config.batch_size)
            eps_start_time = time.time()
            for mb_q, mb_qseq, mb_qmask, mb_a, mb_aseq, mb_amask, mb_neg, mb_negseq, mb_negmask in train_epoch_iter:
                step = step + 1
                #print mbs, batch_xs_seq
                if (mb_q.shape[0] != config.batch_size):
                    print "batch size is different"

                # Fit training using batch data
                sess.run(mtrain.optimizer, feed_dict={mtrain._Q: mb_q, mtrain._Qseq: mb_qseq, mtrain._Qmask: mb_qmask,
                                                      mtrain._A: mb_a, mtrain._Aseq: mb_aseq, mtrain._Amask: mb_amask,
                                                      mtrain._Neg: mb_neg, mtrain._Negseq: mb_negseq, mtrain._Negmask: mb_negmask,})


                if  step % config.display_step == 0:
                    # Calculate batch accuracy and loss
                    [loss, vars] = sess.run([mtrain.cost, mtrain.vars],
                                            feed_dict={mtrain._Q: mb_q, mtrain._Qseq: mb_qseq, mtrain._Qmask: mb_qmask,
                                                      mtrain._A: mb_a, mtrain._Aseq: mb_aseq, mtrain._Amask: mb_amask,
                                                      mtrain._Neg: mb_neg, mtrain._Negseq: mb_negseq, mtrain._Negmask: mb_negmask,})

                    infos = "EP "+str(eps+1)+", MB "+str(mbs)+", lr "+str(config.learning_rate)+", MB Loss= "+ \
                            "{:.4f}".format(loss)
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
                answers = DictToList(ds.Answers)

                valid_epoch_iter = ds.test_iterator(valid_data, config.evaluate_batch_size)
                acc_valid = evaluate_acc(sess, config, mtest, valid_epoch_iter, answers, ds, shelllog)

                if not config.speed_up:
                    test1_epoch_iter = ds.test_iterator(test1_data, config.evaluate_batch_size)
                    acc_test1 = evaluate_acc(sess, config, mtest, test1_epoch_iter, answers, ds, shelllog)

                    train_for_test_epoch_iter = ds.test_iterator(train_data_for_test, config.evaluate_batch_size)
                    acc_train = evaluate_acc(sess, config, mtest, train_for_test_epoch_iter, answers, ds, shelllog)

                    test2_epoch_iter = ds.test_iterator(test2_data, config.evaluate_batch_size)
                    acc_test2 = evaluate_acc(sess, config, mtest, test2_epoch_iter, answers, ds, shelllog)
                else:
                    acc_test2 = acc_test1 = acc_train = acc_valid

                plt_x.append(eps)
                plt_y.append(acc_train)
                plt_y2.append(acc_valid)
                plt_y3.append(acc_test1)
                plt_y4.append(acc_test2)

                print "Valid Accuracy: ", "{:.4f}".format(acc_valid), "Test1 Accuracy: ", "{:.4f}".format(acc_test1)
                print "Train Accuracy: ", "{:.4f}".format(acc_train), "Test2 Accuracy: ", "{:.4f}".format(acc_test2)
                shelllog.write("Validation Accuracy: "+"{:.4f}".format(acc_valid)+
                               " Test1 Accuracy: "+"{:.4f}".format(acc_test1)+"\n")
                shelllog.write("Train Accuracy: "+ "{:.4f}".format(acc_train)+
                               " Test2 Accuracy: "+"{:.4f}".format(acc_test2)+"\n\n")
                print "\n"

                draw_figure(config)

                if config.save_model and acc_valid >= config.best_ans:
                    config.best_ans = acc_valid
                    saver = tf.train.Saver()
                    config.save_path = saver.save(sess, config.save_path)

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

        answers = DictToList(ds.Answers)

        valid_epoch_iter = ds.test_iterator(valid_data, config.evaluate_batch_size)
        acc_valid = evaluate_acc(sess, config, mtest, valid_epoch_iter, answers, ds)

        test1_epoch_iter = ds.test_iterator(test1_data, config.evaluate_batch_size)
        acc_test1 = evaluate_acc(sess, config, mtest, test1_epoch_iter, answers, ds, check=True)

        train_for_test_epoch_iter = ds.test_iterator(train_data_for_test, config.evaluate_batch_size)
        acc_train = evaluate_acc(sess, config, mtest, train_for_test_epoch_iter, answers, ds)

        test2_epoch_iter = ds.test_iterator(test2_data, config.evaluate_batch_size)
        acc_test2 = evaluate_acc(sess, config, mtest, test2_epoch_iter, answers, ds, check=True)

        print "Valid Accuracy: ", "{:.4f}".format(acc_valid), "Test1 Accuracy: ", "{:.4f}".format(acc_test1)
        print "Train Accuracy: ", "{:.4f}".format(acc_train), "Test2 Accuracy: ", "{:.4f}".format(acc_test2)

        shelllog.write("Validation Accuracy: "+"{:.4f}".format(acc_valid)+
                               " Test1 Accuracy: "+"{:.4f}".format(acc_test1)+"\n")
        shelllog.write("Train Accuracy: "+ "{:.4f}".format(acc_train)+
                               " Test2 Accuracy: "+"{:.4f}".format(acc_test2)+"\n\n")

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
