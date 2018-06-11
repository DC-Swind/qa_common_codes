import numpy as np
import time

class Insurance_reader:
    def __init__(self, path ="data/insuranceQA-master/", def_len = 200, def_len_Q=None, def_len_A=None, dim_input = 100, dim_hidden = 141):
        # setting
        self.def_len = def_len
        self.def_len_Q = def_len_Q
        self.def_len_A = def_len_A
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.path = path

        if def_len_A is None or def_len_Q is None:
            self.def_len_A = def_len
            self.def_len_Q = def_len
            
        # from reading
        #self.Answers = self.inputAnswer()
        #self.Vocab = self.inputVocabFile()
        #self.Embeddings = self.inputEmbeddings()
        self.overlap = {}


    def inputEmbeddings(self, file="data/glove.6B.100d.txt"):
        f = open(file, 'r+')
        if "glove" in file:
            pass
        else:
            _ = f.readline()

        lines = f.readlines()
        Embeddings = {}
        for line in lines:
            sp = line.split(" ")
            word = sp[0]
            embed = np.zeros(self.dim_input)
            for i in range(self.dim_input):
                embed[i] = float(sp[i+1])

            Embeddings[word] = embed

        return Embeddings

    def embeddingInit(self):
        # eI[0,:] for unk, self.Vocab start from 1
        eI = np.zeros((len(self.Vocab)+1, self.dim_input))
        c = 0
        for k,v in self.Vocab.items():
            if self.Embeddings.get(v) is not None:
                eI[k,:] = self.Embeddings[v]
                eI[0,:] += eI[k,:]
                c += 1
            else:
                if c != 0:
                    eI[k,:] = eI[0,:] / c
                else:
                    eI[k,:] = np.random.uniform(-0.1, 0.1, size=(self.dim_input))

        eI[0,:] /= len(self.Vocab)
        return eI

    def analysis_qa(self, Q, groundtruths, candidates):
        
        for p in groundtruths:
            A = self.Answers[p]
            count = 0
            for qw in Q:
                for aw in A:
                    if qw == aw:
                        count += 1
                        break
            
            if self.overlap.get(str(Q)) is None:
                self.overlap[str(Q)] = ([], [])
        
            self.overlap[str(Q)][0].append(count)
        
        for n in candidates:
            A = self.Answers[n]
            count = 0
            for qw in Q:
                for aw in A:
                    if qw == aw:
                        count += 1
                        break
            self.overlap[str(Q)][1].append(count)
    
    def print_analysis_qa(self):
        most_overlap = 0
        f = open(self.path + "overlap_count", "w+")
        for k, v in self.overlap.items():
            f.write(k+"\n")
            p = sorted(v[0], reverse = True)
            n = sorted(v[1], reverse = True)
            try:
                if p[0] == n[0]:
                    most_overlap += 1
            except:
                pass
            f.write(str(p)+"\n")
            f.write(str(n)+"\n\n")
            
        f.close()
        print "most overlap can chose", most_overlap, "correct answer, total", len(self.overlap)
        exit()
    
    def format_sentence(self, sentence):
        q = sentence.split(" ")
        return [int(item.split("_")[1]) for item in q]

    def sampleCandidates(self, st, ed, groundTruths, sampleN = 500):
        idx = st + np.arange(ed-st, dtype="int32")
        np.random.shuffle(idx)
        candidates = []
        candidates.extend(groundTruths)
        for index in idx:
            if index not in groundTruths:
                candidates.append(index)
            if len(candidates) == sampleN:
                break
        """
        start = time.time()
        idx = st + np.arange(ed-st, dtype="int32")
        idx = np.array(list(set(idx).difference(set(groundTruths))))
        np.random.shuffle(idx)
        candidates = []
        candidates.extend(groundTruths)
        candidates.extend(idx[0:sampleN-len(groundTruths)])
        print "cost::", time.time() - start
        """
        return candidates

    def inputTrainData(self, file = "question.train.token_idx.label"):
        f = open(self.path+file, 'r+')
        lines= f.readlines()
        # Because in train set one question has serveral answers, so we split them into several QA pairs.
        # lines = 12887, change training data to one answer is 18540
        dataset = {}
        dataset["Question"] = []
        dataset["Answer"] = []
        dataset["GroundTruths"] = []

        # format training data to the format likes test set, sample N (500) candidates.
        datasetForTest = {}
        datasetForTest["Question"] = []
        datasetForTest["GroundTruths"] = []
        datasetForTest["Candidates"] = []

        index = 0
        for line in lines:
            sp = line.split("\t")
            question = sp[0]
            answers = sp[1].replace("\n","").replace("\r","").split(" ")
            # answer index is from 1, change to 0
            answers = [int(ans)-1 for ans in answers]
            Q = self.format_sentence(question)

            datasetForTest["Question"].append(Q)
            datasetForTest["GroundTruths"].append(answers)
            datasetForTest["Candidates"].append(self.sampleCandidates(0, len(self.Answers), answers))

            for ans in answers:
                A = ans
                dataset["Question"].append(Q)
                dataset["Answer"].append(A)
                dataset["GroundTruths"].append(answers)

                index += 1
        f.close()
        dataset["Question"] = np.array(dataset["Question"])
        dataset["Answer"] = np.array(dataset["Answer"])
        dataset["GroundTruths"] = np.array(dataset["GroundTruths"])

        datasetForTest["Question"] = np.array(datasetForTest["Question"])
        datasetForTest["GroundTruths"] = np.array(datasetForTest["GroundTruths"])
        datasetForTest["Candidates"] = np.array(datasetForTest["Candidates"])

        print "inputTrainData:", index
        return dataset, datasetForTest

    def inputTestData(self, file = "question.dev.label.token_idx.pool", isDev = False):
        # dataset {Question: [q1, q2, ...], GroundTruths: [gt1, gt2, ...], Candidates: [c1, c2, ...]}
        # where q1 is [w1, w2, ...], gt1 is [ansID1, ansID2, ...], c1 is [ansID1, ansID2, ...]
        f = open(self.path+file, 'r+')
        lines = f.readlines()
        dataset = {}
        dataset["Question"] = []
        dataset["GroundTruths"] = []
        dataset["Candidates"] = []
        index = 0
        for line in lines:
            sp = line.split("\t")
            Q = self.format_sentence(sp[1])

            groundtruths = sp[0].replace("\n","").replace("\r","").split(" ")
            candidates = sp[2].replace("\n","").replace("\r","").split(" ")
            groundtruths = [int(ans) -1 for ans in groundtruths]
            candidates = [int(ans)-1 for ans in candidates]
            
            self.analysis_qa(Q, groundtruths, candidates)
            dataset["Question"].append(Q)
            dataset["GroundTruths"].append(groundtruths)
            dataset["Candidates"].append(candidates)

            index += 1

        f.close()
        dataset["Question"] = np.array(dataset["Question"])
        dataset["GroundTruths"] = np.array(dataset["GroundTruths"])
        dataset["Candidates"] = np.array(dataset["Candidates"])

        print "input ",file,"number:",index
        return dataset

    def inputAnswer(self, file="answers.label.token_idx"):
        # Answers {index: [w1, w2, ...]}
        f = open(self.path+file, 'r+')
        lines = f.readlines()
        Answers = {}

        for line in lines:
            sp = line.split("\t")
            # ID is from 1, change to 0
            ID = int(sp[0]) - 1
            Sentence = self.format_sentence(sp[1])
            Answers[ID] = Sentence

        f.close()
        print "Answers Number", len(Answers)
        return Answers

    def inputVocabFile(self, file="vocabulary"):
        # Vocab {index: word}
        f = open(self.path+file, 'r+')
        lines = f.readlines()

        Vocab = {}

        for line in lines:
            sp = line.split("\t")
            Vocab[int(sp[0].split("_")[1])] = sp[1].replace("\n","").replace("\r","")

        return Vocab

    def convert_index_to_word(self, indexes):
        #indexes is (time)
        sent = []
        for i in range(indexes.shape[0]):
            if indexes[i] == 0:
                break
            sent.append(self.Vocab[indexes[i]])
        return sent

    def outputInfo(self, Trainset, Devset, Test1set, Test2set, TrainForTest):
        print "Train dataset contains", len(Trainset["Question"]), "QA pairs."
        print "TrainForTest contains", len(TrainForTest["Question"]), "Questions."
        print "Dev dataset contains", len(Devset["Question"]), "Questions."
        print "Test1 dataset contains", len(Test1set["Question"]), "Questions."
        print "Test2 dataset contains", len(Test2set["Question"]), "Questions."
        print len(self.Answers),"answers in total."
        print "Vocabulary size is", len(self.Vocab)
        count = 0
        for k,v in self.Vocab.items():
            if self.Embeddings.get(v) is not None:
                count += 1
        print count,"words have embeddings."

    def sampleForTest(self, dataset):
        total = len(dataset["Question"])
        sample = total // 10
        idx = np.arange(total, dtype="int32")
        np.random.shuffle(idx)
        idx = idx[:sample]
        subset = {}
        subset["Question"] = dataset["Question"][idx]
        subset["GroundTruths"] = dataset["GroundTruths"][idx]
        subset["Candidates"] = dataset["Candidates"][idx]
        return subset

    def loading(self):
        self.Answers = self.inputAnswer()
        TrainSet, TrainSetForTest = self.inputTrainData()
        DevSet = self.inputTestData(file="question.dev.label.token_idx.pool", isDev=True)
        Test1Set = self.inputTestData(file="question.test1.label.token_idx.pool", isDev=False)
        Test2Set = self.inputTestData(file="question.test2.label.token_idx.pool", isDev=False)
        self.Vocab = self.inputVocabFile()
        #self.print_analysis_qa()
        # default using glove.100d, try to use pretrain 100d word2vec
        #self.Embeddings = self.inputEmbeddings()
        if self.dim_input == 100:
            self.Embeddings = self.inputEmbeddings(file = self.path + "insurance_w2v_100d_pretrain.output")
        elif self.dim_input == 300:
            #self.Embeddings = self.inputEmbeddings(file="data/glove.840B.300d.txt")
            self.Embeddings = self.inputEmbeddings(file=self.path + "insurance_w2v_300d_pretrain.bin")
        elif self.dim_input == 500:
            self.Embeddings = self.inputEmbeddings(file=self.path + "insurance_w2v_500d_pretrain_50iter.bin")
        TrainForTest = self.sampleForTest(TrainSetForTest)
        self.outputInfo(TrainSet, DevSet, Test1Set, Test2Set, TrainForTest)
        return TrainSet, DevSet, Test1Set, Test2Set, TrainForTest


    def epoch_iterator(self, dataset, batch_size):
        # dataset is an dict, dataset["Question"], dataset["Answer"], dataset["GroundTruths"]
        # "Question" is an array, whose item is a sentence which is a list of word indexes.
        # "Answer" is an array, whose item is an answer index.
        # "GroundTruths" is an array, whose item is a list of answer indexes, which are all answers belong to this question.

        idx = np.arange(len(dataset["Question"]), dtype="int32")
        np.random.shuffle(idx)

        epoch_size = len(idx) // batch_size

        for i in range(epoch_size):
            Q = np.zeros(shape=[batch_size, self.def_len_Q], dtype=np.int32)
            A = np.zeros(shape=[batch_size, self.def_len_A], dtype=np.int32)
            Neg = np.zeros(shape=[batch_size, self.def_len_A], dtype=np.int32)

            maskQ = np.zeros([batch_size, self.def_len_Q, 1], dtype=np.float32)
            maskA = np.zeros([batch_size, self.def_len_A, 1], dtype=np.float32)
            maskNeg = np.zeros([batch_size, self.def_len_A, 1], dtype=np.float32)

            seqQ = np.zeros([batch_size], dtype=np.int32)
            seqA = np.zeros([batch_size], dtype=np.int32)
            seqNeg = np.zeros([batch_size], dtype=np.int32)

            mbQ = dataset["Question"][idx[i*batch_size : (i+1)*batch_size]]
            mbA = dataset["Answer"][idx[i*batch_size : (i+1)*batch_size]]
            mbG = dataset["GroundTruths"][idx[i*batch_size : (i+1)*batch_size]]

            for j in range(batch_size):
                q = mbQ[j]
                a = mbA[j]
                g = mbG[j]
                while(True):
                    neg = np.random.randint(0, len(self.Answers))
                    if neg not in g:
                        break
                a = self.Answers[a]
                neg = self.Answers[neg]

                Q[j,:], maskQ[j,:,:], seqQ[j] = self.clip_sentence(q, maskQ[0,:,:], isQuestion=True)
                A[j,:], maskA[j,:,:], seqA[j] = self.clip_sentence(a, maskA[0,:,:], isQuestion=False)
                Neg[j,:], maskNeg[j,:,:], seqNeg[j] = self.clip_sentence(neg, maskA[0,:,:], isQuestion=False)

            yield (Q, seqQ, maskQ, A, seqA, maskA, Neg, seqNeg, maskNeg)

        if epoch_size * batch_size < len(idx):
            small_batch = len(idx) - epoch_size * batch_size

            Q = np.zeros(shape=[small_batch, self.def_len_Q], dtype=np.int32)
            A = np.zeros(shape=[small_batch, self.def_len_A], dtype=np.int32)
            Neg = np.zeros(shape=[small_batch, self.def_len_A], dtype=np.int32)

            maskQ = np.zeros([small_batch, self.def_len_Q, 1], dtype=np.float32)
            maskA = np.zeros([small_batch, self.def_len_A, 1], dtype=np.float32)
            maskNeg = np.zeros([small_batch, self.def_len_A, 1], dtype=np.float32)

            seqQ = np.zeros([small_batch], dtype=np.int32)
            seqA = np.zeros([small_batch], dtype=np.int32)
            seqNeg = np.zeros([small_batch], dtype=np.int32)

            mbQ = dataset["Question"][idx[epoch_size*batch_size : epoch_size*batch_size+small_batch]]
            mbA = dataset["Answer"][idx[epoch_size*batch_size : epoch_size*batch_size+small_batch]]
            mbG = dataset["GroundTruths"][idx[epoch_size*batch_size : epoch_size*batch_size+small_batch]]

            for j in range(small_batch):
                q = mbQ[j]
                a = mbA[j]
                g = mbG[j]
                while(True):
                    neg = np.random.randint(0, len(self.Answers))
                    if neg not in g:
                        break
                a = self.Answers[a]
                neg = self.Answers[neg]

                Q[j,:], maskQ[j,:,:], seqQ[j] = self.clip_sentence(q, maskQ[0,:,:], isQuestion=True)
                A[j,:], maskA[j,:,:], seqA[j] = self.clip_sentence(a, maskA[0,:,:], isQuestion=False)
                Neg[j,:], maskNeg[j,:,:], seqNeg[j] = self.clip_sentence(neg, maskA[0,:,:], isQuestion=False)

            yield (Q, seqQ, maskQ, A, seqA, maskA, Neg, seqNeg, maskNeg)

    def test_Answer_iterator(self, batch_size, Answers = None):
        # prepare for generate embedding of all answers
        if Answers is None:
            Answers = self.Answers

        epoch_size = len(Answers) // batch_size

        for i in range(epoch_size):
            A = np.zeros(shape=[batch_size, self.def_len_A], dtype=np.int32)
            maskA = np.zeros([batch_size, self.def_len_A, 1], dtype=np.float32)
            seqA = np.zeros([batch_size], dtype=np.int32)

            for j in range(batch_size):
                a = Answers[i * batch_size + j]
                A[j,:], maskA[j,:,:], seqA[j] = self.clip_sentence(a, maskA[0,:,:], isQuestion=False)

            yield (A, seqA, maskA)

        if epoch_size * batch_size < len(Answers):
            small_batch = len(Answers) - epoch_size * batch_size
            A = np.zeros(shape=[small_batch, self.def_len_A], dtype=np.int32)
            maskA = np.zeros([small_batch, self.def_len_A, 1], dtype=np.float32)
            seqA = np.zeros([small_batch], dtype=np.int32)

            for j in range(small_batch):
                a = Answers[epoch_size * batch_size + j]
                A[j,:], maskA[j,:,:], seqA[j] = self.clip_sentence(a, maskA[0,:,:], isQuestion=False)

            yield (A, seqA, maskA)

    def clip_sentence(self, s, maskshape, isQuestion=True):
        if isQuestion:
            def_len = self.def_len_Q
        else:
            def_len = self.def_len_A

        seq = min(len(s), def_len)
        S = np.zeros(shape=[def_len], dtype=np.int32)
        mask = np.zeros_like(maskshape)
        for k in range(seq):
            S[k] = s[k]
            mask[k] += 1
        return S, mask, seq

    def test_iterator(self, dataset, batch_size):
        # dataset is an dict, dataset["Question"], dataset["Candidates"], dataset["GroundTruths"]
        # "Question" is an array, whose item is a sentence which is a list of word indexes.
        # "Candidates" is an array, whose item is a list of answer indexes, which are all answers for choosing.
        # "GroundTruths" is an array, whose item is a list of answer indexes, which are all answers belong to this question.

        idx = np.arange(len(dataset["Question"]), dtype="int32")
        #np.random.shuffle(idx)

        epoch_size = len(idx) // batch_size

        for i in range(epoch_size):
            Q = np.zeros(shape=[batch_size, self.def_len_Q], dtype=np.int32)
            maskQ = np.zeros([batch_size, self.def_len_Q, 1], dtype=np.float32)
            seqQ = np.zeros([batch_size], dtype=np.int32)

            mbQ = dataset["Question"][idx[i*batch_size : (i+1)*batch_size]]
            mbG = dataset["GroundTruths"][idx[i*batch_size : (i+1)*batch_size]]
            mbC = dataset["Candidates"][idx[i*batch_size : (i+1)*batch_size]]

            for j in range(batch_size):
                q = mbQ[j]
                Q[j,:], maskQ[j,:,:], seqQ[j] = self.clip_sentence(q, maskQ[0,:,:], isQuestion=True)

            yield (Q, seqQ, maskQ, mbG, mbC)

        if epoch_size * batch_size < len(idx):
            small_batch = len(idx) - epoch_size * batch_size

            Q = np.zeros(shape=[small_batch, self.def_len_Q], dtype=np.int32)
            maskQ = np.zeros([small_batch, self.def_len_Q, 1], dtype=np.float32)
            seqQ = np.zeros([small_batch], dtype=np.int32)

            mbQ = dataset["Question"][idx[epoch_size*batch_size : epoch_size*batch_size+small_batch]]
            mbG = dataset["GroundTruths"][idx[epoch_size*batch_size : epoch_size*batch_size+small_batch]]
            mbC = dataset["Candidates"][idx[epoch_size*batch_size : epoch_size*batch_size+small_batch]]

            for j in range(small_batch):
                q = mbQ[j]
                Q[j,:], maskQ[j,:,:], seqQ[j] = self.clip_sentence(q, maskQ[0,:,:], isQuestion=True)

            yield (Q, seqQ, maskQ, mbG, mbC)


    def convert_train_file(self, file = "question.train.token_idx.label"):
        f = open(self.path+file, 'r+')
        lines= f.readlines()

        Q = []
        for line in lines:
            sp = line.split("\t")
            question = sp[0]

            q = self.format_sentence(question)
            Q.append(q)

        f.close()
        return Q

    def convert_dev_file(self, file):
        f = open(self.path+file, 'r+')
        lines = f.readlines()

        Q = []
        for line in lines:
            sp = line.split("\t")
            q = self.format_sentence(sp[1])
            Q.append(q)

        f.close()
        return Q



    def convert_qa_to_words(self, useTest = False, output_filename = "text_for_w2v.txt"):
        # convert questions and answers to word from index
        # save to output_filename, one question or answer each line
        # use this file to train 100d word2vec for word embedding init

        # not use test file, but I don't know whether "LSTM-based.." and "Inner .." use or not.
        devfile= "question.dev.label.token_idx.pool"
        test1file = "question.test1.label.token_idx.pool"
        test2file = "question.test2.label.token_idx.pool"

        save_file = open(self.path + output_filename, "w+")

        Vocab = self.inputVocabFile()
        answers = self.inputAnswer()
        for id, a in answers.iteritems():
            for word_idx in a:
                save_file.write(Vocab[word_idx] + " ")
            save_file.write("<eos>\n")

        questions_train = self.convert_train_file()
        for q in questions_train:
            for word_idx in q:
                save_file.write(Vocab[word_idx] + " ")
            save_file.write("<eos>\n")

        questions_dev = self.convert_dev_file(file = devfile)
        for q in questions_dev:
            for word_idx in q:
                save_file.write(Vocab[word_idx] + " ")
            save_file.write("<eos>\n")

        if useTest:
            questions_test1 = self.convert_dev_file(file = test1file)
            for q in questions_test1:
                for word_idx in q:
                    save_file.write(Vocab[word_idx] + " ")
                save_file.write("<eos>\n")

            questions_test2 = self.convert_dev_file(file = test2file)
            for q in questions_test2:
                for word_idx in q:
                    save_file.write(Vocab[word_idx] + " ")
                save_file.write("<eos>\n")

        save_file.close()


""" Codes for convert qa to words """
"""
ds = Insurance_reader(path="data/insuranceQA-master/")
ds.convert_qa_to_words(useTest=False, output_filename="text_for_w2v.txt")
"""
