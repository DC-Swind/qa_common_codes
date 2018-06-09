# coding: utf-8
import os
import time
import numpy as np
import pandas as pd
import re
import xml.dom.minidom
import struct

def replace_stop(x):

    x = x.replace("<br />"," ").replace("<br>"," ").replace("$", " $ ").replace("%"," % ")
    x = x.replace("´","'").replace("\n"," ").replace("”"," ")
    x = x.replace("·", " ").replace("°"," ").replace("º"," ").replace("…"," ").replace("“"," ").replace("|"," ")
    x = x.replace("’", "'").replace("‘"," ").replace("♥"," ").replace("¿"," ").replace("®"," ").replace("—"," ")
    x = x.replace("'m", " am").replace("'re"," are").replace("'ll", " will")
    x = x.replace("won't", "will not").replace("n't", " not")
    x = x.replace("\xc2\xa0", " ")

    re_delete_space = re.compile("[ ]+")
    x, number = re_delete_space.subn(" ",x)

    re_dollar1 = re.compile("(?<=\$) (?=[0-9]+)")
    x, _ = re_dollar1.subn("", x)
    re_dollar2 = re.compile("(?<=[0-9]) (?=[$%])")
    x, _ = re_dollar2.subn("", x)
    re_num1 = re.compile("(?<![0-9])[.:,-]")
    x, _ = re_num1.subn(" ", x)
    re_num2 = re.compile("[.:,-](?![0-9]+)")
    x, _ = re_num2.subn(" ", x)

    # can drop 's
    drop_s = True
    if drop_s:
        re_s = re.compile("\'s")
        x, _ = re_s.subn("", x)
        re_quote = re.compile("\'")
        x, _ = re_quote.subn(" ", x)
    else:
        re_s = re.compile(" (?=\'s)")
        x, _ = re_s.subn("", x)
        re_quote = re.compile("\'(?!s)")
        x, _ = re_quote.subn(" ", x)

    re_to_space = "/?!<>~&#_\";*()`{}[]\\^+"
    for c in re_to_space:
        x = x.replace(c, " ")
    x = x.replace("="," ").replace("``", " ")

    #re_delete_number = re.compile("[0-9]+")
    #x, number = re_delete_number.subn(" ",x)
    re_delete_space = re.compile("[ ]+")
    x, number = re_delete_space.subn(" ",x)
    return x


class yahoo_reader:
    def __init__(self, path = "data/", deflen =200, n_input = 300):
        self.path = path
        self.vocab = {"<unk>":0}  # word to index
        self.reverse_vocab = {0:"<unk>"}
        self.unk_in_test = {}
        self.vocab_in_test = {}
        self.question_unk = {}
        self.question_vocab = {}
        self.embeddings = {} # word index to embedding
        self.deflen = deflen
        self.dim_input = n_input

        self.sentence_max_len = 0
        self.answer_min_len = 999999
        self.sentence_num = 0
        self.longer_than_def_count = 0
        self.answer_length_sum = 0
        self.answer_num = 0

    def convert_indexes_to_sent(self, sent_indexes):
        sent = ""
        for idx in sent_indexes:
            sent = sent + self.reverse_vocab[idx]+" "
        return sent

    def splitSentence(self, sent, add=False, dataset=None, isAnswer=False):
        #print sent.__class__
        #sent = sent.encode("ascii", errors="replace")
        #print sent.__class__
        if not isinstance(sent, str):
            sent = sent.encode("utf-8", errors="ignore")
        #print sent.__class__

        sent = replace_stop(sent)

        sp = sent.split()
        rt = []
        for itemB in sp:
            item = itemB.lower()
            if item == "":
                continue

            if dataset is not None and dataset == "test":
                self.vocab_in_test[item] = 1
            if self.vocab.get(item) is not None:
                rt.append(self.vocab[item])
            else:
                if add:
                    self.vocab[item] = len(self.vocab)
                    self.reverse_vocab[self.vocab[item]] = item
                    rt.append(self.vocab[item])
                else:
                    rt.append(self.vocab["<unk>"])
                    if dataset is not None and dataset == "test":
                        self.unk_in_test[item] = 1

            if not add and not isAnswer:
                self.question_vocab[item] = 1
                if self.vocab.get(item) is None:
                    self.question_unk[item] = 1

        if len(rt) > self.sentence_max_len:
            self.sentence_max_len = len(rt)
        if isAnswer:
            if len(rt) < self.answer_min_len:
                self.answer_min_len = len(rt)

        self.sentence_num += 1
        if len(rt) > self.deflen:
            self.longer_than_def_count += 1

        if isAnswer:
            self.answer_length_sum += len(rt)
            self.answer_num += 1
        return rt

    def save_vocab_and_unk(self):
        f = open(self.path + "vocab.txt", "w+")
        for k, v in self.vocab.items():
            f.write(k+"\n")
        f.close()

        f = open(self.path + "unk_in_test.txt", "w+")
        for k, v in self.unk_in_test.items():
            f.write(k+"\n")
        f.close()

    def check_positive(self, sentence, isQ=False):
        sent = sentence.encode("utf-8", errors="ignore")
        sent = replace_stop(sent)
        #f = open("clean-text.txt", "a+")
        #f.write(sent+"\n")
        #f.close()
        s = sent.split()

        if isQ:
            if len(s) >= 5 and len(s) <= 50:
                return True
        else:
            if len(s) >= 5 and len(s) <= 50:
                return True

        return False

    def readfile(self, filename, output=False, version=""):
        cpt = time.time()

        dom = xml.dom.minidom.parse(self.path + filename)
        Q = []
        P = []
        N = []
        C = []
        categary = {}
        ystfeed = dom.documentElement
        #print "ystfeed", ystfeed.nodeName
        vespaadds = ystfeed.getElementsByTagName('vespaadd')
        question_num = len(vespaadds)
        print "Has", question_num, "questions in total"
        for i in range(question_num):
            vespaadd = vespaadds[i]
            #print "vespaadd", vespaadd.nodeName
            document = vespaadd.getElementsByTagName('document')[0]
            #print "document", document.nodeName

            uri = document.getElementsByTagName('uri')[0]
            #print "uri", uri.nodeName, uri.firstChild.data

            subject = document.getElementsByTagName('subject')[0]
            #print "subject", subject.nodeName, subject.firstChild.data

            #content = document.getElementsByTagName('content')[0]
            #print "content", content.nodeName, content.firstChild.data

            bestanswers = document.getElementsByTagName('bestanswer')
            if len(bestanswers) > 1:
                print "has more than one best answers.", len(bestanswers)
            bestanswer = bestanswers[0]
            #print "bestanswer", bestanswer.nodeName, bestanswer.firstChild.data

            maincat = document.getElementsByTagName('maincat')
            cat = None
            if len(maincat) > 0:
                cat = maincat[0].firstChild.data
                cat = cat.encode("utf-8", errors="ignore")
                if categary.get(cat) is None:
                    categary[cat] = [subject.firstChild.data]
                else:
                    categary[cat].append(subject.firstChild.data)
                #if cat == "Business & Finance":
                #if cat == "Family & Relationships": #"
                """
                if cat == "Health" or cat == "Family & Relationships" or cat == "Business & Finance":
                    pass
                else:
                    cat = None
                """
            nbestanswers = document.getElementsByTagName('nbestanswers')[0]
            #print "nbestanswers", nbestanswers.nodeName
            answer_items = nbestanswers.getElementsByTagName('answer_item')
            items_num = len(answer_items)
            N_tmp = []
            #print "This question has", items_num, "negtive answers"
            for j in range(items_num):
                answer_item = answer_items[j]
                if self.check_positive(answer_item.firstChild.data):
                    N_tmp.append(answer_item.firstChild.data)
                #print answer_item.nodeName, answer_item.firstChild.data.encode(encoding="utf-8", errors="ignore")

            if self.check_positive(bestanswer.firstChild.data) and \
                    self.check_positive(subject.firstChild.data, isQ = True) and cat is not None:
                Q.append(subject.firstChild.data)
                P.append(bestanswer.firstChild.data)
                N.append(N_tmp)
                C.append(cat)



        print "Has", len(Q), "questions in total"
        print "Time:", time.time() - cpt

        Q, P, N, C = np.array(Q), np.array(P), np.array(N), np.array(C)
        idx = np.arange(len(Q), dtype="int32")
        np.random.shuffle(idx)

        split1 = int(0.8*len(Q))
        split2 = int(0.9*len(Q))
        train_idx = idx[0:split1]
        dev_idx = idx[split1:split2]
        test_idx = idx[split2:]
        Qtrain, Qdev, Qtest = Q[train_idx], Q[dev_idx], Q[test_idx]
        Ptrain, Pdev, Ptest = P[train_idx], P[dev_idx], P[test_idx]
        Ntrain, Ndev, Ntest = N[train_idx], N[dev_idx], N[test_idx]
        Ctrain, Cdev, Ctest = C[train_idx], C[dev_idx], C[test_idx]

        Answer_Pool = []
        def format_data(Q, P, N, C, add, dataset=None):
            q = []
            p = []
            n = []
            for i in range(len(Q)):
                q.append(self.splitSentence(Q[i], add=add, dataset=dataset))
                p.append(self.splitSentence(P[i], add=add, dataset=dataset, isAnswer=True))
                n_tmp = []
                for j in range(len(N[i])):
                    tmp = self.splitSentence(N[i][j], add=add, dataset=dataset, isAnswer=True)
                    n_tmp.append(tmp)
                    Answer_Pool.append(tmp)
                n.append(n_tmp)
            return q, p, n, C

        Qtrain, Ptrain, Ntrain, Ctrain = format_data(Qtrain, Ptrain, Ntrain, Ctrain, add=True)
        Qdev, Pdev, Ndev, Cdev = format_data(Qdev, Pdev, Ndev, Cdev, add=True)
        Qtest, Ptest, Ntest, Ctest = format_data(Qtest, Ptest, Ntest, Ctest, add=True, dataset="test")

        print "vocab:", len(self.vocab), "vocab in test:", len(self.vocab_in_test), "unk in test:", len(self.unk_in_test)
        print "longer than def:", self.longer_than_def_count, "max length:", self.sentence_max_len
        print "answer_num:", self.answer_num, "avg length of answer:", float(self.answer_length_sum)/self.answer_num
        #self.save_vocab_and_unk()

        def span_negtive(N, P, sampleN =4):
            for i in range(len(N)):
                p = P[i]
                for j in range(len(N[i])):
                    n = N[i][j]
                    if p == n:
                        del N[i][j]
                        break
                n_neg = len(N[i])
                if n_neg > sampleN:
                    N[i] = N[i][0:sampleN]
                if n_neg == sampleN:
                    continue
                for j in range(sampleN - n_neg):
                    r = np.random.randint(0, len(Answer_Pool))
                    N[i].append(Answer_Pool[r])
            return N

        #Ntrain = span_negtive(Ntrain, Ptrain)
        #Ndev = span_negtive(Ndev, Pdev)
        #Ntest = span_negtive(Ntest, Ptest)

        Train = (np.array(Qtrain), np.array(Ptrain), np.array(Ntrain), Ctrain)
        Dev = (np.array(Qdev), np.array(Pdev), np.array(Ndev), Cdev)
        Test = (np.array(Qtest), np.array(Ptest), np.array(Ntest), Ctest)

        def output_dataset(Q, P, N, C, dataset, output_negtive=True):
            f = open(self.path+"yahooQA-"+dataset+".txt", "w+")
            for i in range(len(Q)):
                q = self.convert_indexes_to_sent(Q[i])
                p = self.convert_indexes_to_sent(P[i])
                f.write(C[i]+"\t"+q+"\t"+p+"\t1\n")
                if output_negtive:
                    ns = N[i]
                    for n in ns:
                        f.write(q+"\t"+self.convert_indexes_to_sent(n)+"\t0\n")
            f.close()
        """
        output_dataset(Qtrain, Ptrain, Ntrain, "train")
        output_dataset(Qdev, Pdev, Ndev, "dev")
        output_dataset(Qtest, Ptest, Ntest, "test")
        """
        if output:
            output_dataset(Qtrain, Ptrain, Ntrain, Ctrain, "luence-train-qp", output_negtive=False)
            output_dataset(Qdev, Pdev, Ndev, Cdev, "luence-dev-qp", output_negtive=False)
            output_dataset(Qtest, Ptest, Ntest, Ctest, "luence-test-qp", output_negtive=False)

            f = open(self.path+"yahooQA-answer-pool"+version+".txt", "w+")
            for i in range(len(Answer_Pool)):
                a = self.convert_indexes_to_sent(Answer_Pool[i])
                f.write(a+"\n")
            f.close()
        """
        print "total category:", len(categary)
        for k, v in categary.items():
            print len(v)
        """
        return Train, Dev, Test

    def input_trainset(self, dataset = "train", version=""):
        st = time.time()
        path = self.path + "yahooQA-"+ dataset + "-luence"+version+".txt"

        f = open(path, "r+")
        contents = f.read().split("\n")
        f.close()

        Q = []
        A = []
        N = []
        C = []

        current_q = None
        AN = {}
        hasGroundTruth = False
        for line in contents:
            tmp = line.split("\t")
            if len(tmp) < 3:
                break

            if len(tmp) == 3:
                q, a, label = tmp[0], tmp[1], int(tmp[2])
            else:
                c, q, a, label = tmp[0], tmp[1], tmp[2], int(tmp[3])
            if q != current_q:
                if current_q is not None and hasGroundTruth:
                    for aa in AN[1]:
                        for nn in AN[0]:
                            Q.append(self.splitSentence(current_q, add=True, dataset=dataset))
                            A.append(self.splitSentence(aa, add=True, dataset=dataset, isAnswer=True))
                            N.append(self.splitSentence(nn, add=True, dataset=dataset, isAnswer=True))

                current_q = q
                AN[0] = []
                AN[1] = []
                AN[label].append(a)
                hasGroundTruth = False | label

            else:
                AN[label].append(a)
                hasGroundTruth = hasGroundTruth | label

        print "input cost", time.time() - st,"seconds, Total: ", len(Q), "/", len(contents)
        #print Judge
        return (np.array(Q, dtype=list), np.array(A, dtype=list), np.array(N, dtype=list))


    def input_testset(self, set = "train", add=False, version=""):
        st = time.time()
        path = self.path + "yahooQA-"+ set + "-luence"+version+".txt"
        file = open(path, "r+")
        contents = file.read().split("\n")
        file.close()

        Q = []
        A = []
        current_q = None
        AN = {}
        Judge = []
        GroundTruths = []
        C = []
        for line in contents:
            tmp = line.split("\t")
            if len(tmp) < 3:
                break
            if len(tmp) == 3:
                q, a, label = tmp[0], tmp[1], tmp[2]
                c = "None"
            else:
                c, q, a, label = tmp[0], tmp[1], tmp[2], tmp[3]

            if q != current_q:
                if current_q is not None:
                    for aa in AN[1]:
                        Q.append(self.splitSentence(current_q, add=add, dataset=set))
                        A.append(self.splitSentence(aa, add=add, dataset=set, isAnswer=True))
                    for nn in AN[0]:
                        Q.append(self.splitSentence(current_q, add=add, dataset=set))
                        A.append(self.splitSentence(nn, add=add, dataset=set, isAnswer=True))

                    GroundTruths.append(len(AN[1])) # if answer < GroundTruths[q], then acc += 1
                    Judge.append(len(AN[1]) + len(AN[0]))
                    C.append(c)

                current_q = q
                AN[0] = []
                AN[1] = []
                AN[int(label)].append(a)

            else:
                AN[int(label)].append(a)

        print "input", set, "cost", time.time() - st,"seconds, Total: ", len(Q),"/",len(Judge), "/", len(contents)
        #print Judge
        return (np.array(Q, dtype=list), np.array(A, dtype=list), GroundTruths, Judge, C)


    def inputW2V(self, path = "/home/xud/Music/IMDB/aclImdb/GoogleNews-vectors-negative300.bin",
                       savepath = "data/google-300d-embeddings.txt", dim = 300):
        cpt = time.time()
        f = open(path, "rb")
        [size, dim] = f.readline().split(" ")
        size = int(size)
        dim= int(dim)
        print "has", size, "words, each word has dimensions", dim

        for i in range(size):
            word = ""
            while(True):
                c = f.read(1)
                if c == ' ':
                    break
                else:
                    word += c
            #print word
            vec = np.zeros(dim)
            for j in range(dim):
                vec[j], = struct.unpack('f',f.read(4))
            #print vec

            if self.vocab.get(word) is not None:
                self.embeddings[self.vocab[word]] = vec

        f.close()
        print "total has",len(self.vocab),"words", len(self.embeddings),"has embeddings"," T:[", "{:.2f}".format(time.time()-cpt),"s]"

        print "save embeddings to", savepath
        f = open(savepath, "w+")
        for k, v in self.vocab.items():
            if self.embeddings.get(v) is None:
                f.write(k+" <unk>\n")
            else:
                f.write(k)
                vec = self.embeddings[v]
                for j in range(dim):
                    f.write(" "+str(vec[j]))
                f.write("\n")
        f.close()

    def inputGloVe(self, vocab="glove.6B.100d.txt", output = False):
        # not used in this experiment
        f = open(self.path+vocab, 'r')
        lines = f.readlines()
        for line in lines:
            sp = line.split(" ")
            word = sp[0]
            embed = np.zeros(self.dim_input)
            for i in range(self.dim_input):
                embed[i] = float(sp[i+1])
            if self.vocab.get(word) is not None:
                self.embeddings[self.vocab[word]] = embed
        f.close()
        if output:
            f = open(self.path+"saved_glove_300d_embedding.txt", "w+")
            for k, v in self.embeddings.items():
                word = self.reverse_vocab[k]
                embed = v
                f.write(word)
                for i in range(embed.shape[0]):
                    f.write(" "+str(embed[i]))
                f.write("\n")
            f.close()
            print "output saved vocab embedding to file."
        return self.embeddings

    def inputSavedEmbeddings(self, path = "data/google-300d-embeddings.txt", dim=300):
        f = open(path, "r")
        lines = f.readlines()
        Eunk = np.zeros((dim,))
        unk = 0
        for line in lines:
            sp = line.split(" ")
            word = sp[0]
            if len(sp) < dim:
                #unk
                self.embeddings[self.vocab[word]] = None
                unk += 1
            else:
                vec = np.zeros((dim,))
                for i in range(dim):
                    vec[i] = float(sp[i+1])
                self.embeddings[self.vocab[word]] = vec
                Eunk += vec
        f.close()

        Eunk /= (len(lines) - unk)

        for k, v in self.vocab.items():
            if self.embeddings.get(v) is not None and self.embeddings[v] is None:
                self.embeddings[v] = Eunk
        print "total has",len(lines),"words,  unk:",unk

    def inputEmbeddings(self, file="yahooQA-w2v-100d-100iter-pretrain.embedding"):
        f = open(self.path+file, 'r')
        if "pretrain" in file:
            _ = f.readline()

        lines = f.readlines()
        for line in lines:
            sp = line.split(" ")
            word = sp[0]
            if self.vocab.get(word) is not None:
                embed = np.zeros(self.dim_input)
                for i in range(self.dim_input):
                    embed[i] = float(sp[i+1])

                self.embeddings[self.vocab[word]] = embed
        f.close()
        return self.embeddings

    def embeddingInit(self):
        #f = open(self.path+"unk_embedding", "w+")
        eI = np.zeros((len(self.vocab), self.dim_input))
        for v, k in self.vocab.items():
            if self.embeddings.get(k) is not None:
                eI[k,:] = self.embeddings[k]
            else:
                #print "embeddingInit error."
                #f.write(v+"\n")
                eI[k,:] = np.random.uniform(-0.1, 0.1, size=(self.dim_input))

        #f.close()
        return eI


    def epoch_iterator(self, dataset, batch_size, time_steps = None):

        Q, A, N = dataset
        if time_steps is None:
            time_steps = self.deflen
        idx = np.arange(len(Q), dtype="int32")
        np.random.shuffle(idx)

        epoch_size = len(idx) // batch_size

        for i in range(epoch_size):

            q = np.zeros([batch_size, time_steps], dtype=np.int32)
            a = np.zeros([batch_size, time_steps], dtype=np.int32)
            n = np.zeros([batch_size, time_steps], dtype=np.int32)
            maskq = np.zeros([batch_size, time_steps, 1], dtype=np.float32)
            maska = np.zeros([batch_size, time_steps, 1], dtype=np.float32)
            maskn = np.zeros([batch_size, time_steps, 1], dtype=np.float32)
            seqq = np.zeros([batch_size], dtype=np.int32)
            seqa = np.zeros([batch_size], dtype=np.int32)
            seqn = np.zeros([batch_size], dtype=np.int32)
            mbq = Q[idx[i*batch_size : (i+1)*batch_size]]
            mba = A[idx[i*batch_size : (i+1)*batch_size]]
            mbn = N[idx[i*batch_size : (i+1)*batch_size]]

            for j in range(batch_size):
                qq = mbq[j]
                aa = mba[j]
                nn = mbn[j]
                seqq[j] = min(len(qq), time_steps)
                seqa[j] = min(len(aa), time_steps)
                seqn[j] = min(len(nn), time_steps)
                for k in range(seqq[j]):
                    q[j,k] = qq[k]
                    maskq[j,k,:] += 1
                for k in range(seqa[j]):
                    a[j,k] = aa[k]
                    maska[j,k,:] += 1
                for k in range(seqn[j]):
                    n[j,k] = nn[k]
                    maskn[j,k,:] += 1

            yield (q, seqq, maskq, a, seqa, maska, n, seqn, maskn)

        if epoch_size * batch_size < len(idx):
            small_batch = len(idx) - epoch_size * batch_size
            q = np.zeros([small_batch, time_steps], dtype=np.int32)
            a = np.zeros([small_batch, time_steps], dtype=np.int32)
            n = np.zeros([small_batch, time_steps], dtype=np.int32)
            maskq = np.zeros([small_batch, time_steps, 1], dtype=np.float32)
            maska = np.zeros([small_batch, time_steps, 1], dtype=np.float32)
            maskn = np.zeros([small_batch, time_steps, 1], dtype=np.float32)
            seqq = np.zeros([small_batch], dtype=np.int32)
            seqa = np.zeros([small_batch], dtype=np.int32)
            seqn = np.zeros([small_batch], dtype=np.int32)
            mbq = Q[idx[epoch_size*batch_size : epoch_size*batch_size+small_batch]]
            mba = A[idx[epoch_size*batch_size : epoch_size*batch_size+small_batch]]
            mbn = N[idx[epoch_size*batch_size : epoch_size*batch_size+small_batch]]

            for j in range(small_batch):
                qq = mbq[j]
                aa = mba[j]
                nn = mbn[j]
                seqq[j] = min(len(qq), time_steps)
                seqa[j] = min(len(aa), time_steps)
                seqn[j] = min(len(nn), time_steps)
                for k in range(seqq[j]):
                    q[j,k] = qq[k]
                    maskq[j,k,:] += 1
                for k in range(seqa[j]):
                    a[j,k] = aa[k]
                    maska[j,k,:] += 1
                for k in range(seqn[j]):
                    n[j,k] = nn[k]
                    maskn[j,k,:] += 1

            yield (q, seqq, maskq, a, seqa, maska, n, seqn, maskn)
    """
    def epoch_test_iterator(self, dataset, time_steps = None):
        Q, A, GroundTruths, Jugde = dataset
        if time_steps is None:
            time_steps = self.deflen

        st = 0
        for batch_size, gd in zip(Jugde, GroundTruths):
            q = np.zeros([batch_size, time_steps], dtype=np.int32)
            a = np.zeros([batch_size, time_steps], dtype=np.int32)
            maskq = np.zeros([batch_size, time_steps, 1], dtype=np.float32)
            maska = np.zeros([batch_size, time_steps, 1], dtype=np.float32)
            seqq = np.zeros([batch_size], dtype=np.int32)
            seqa = np.zeros([batch_size], dtype=np.int32)
            mbq = Q[st:st+batch_size]
            mba = A[st:st+batch_size]

            st += batch_size
            for j in range(batch_size):
                qq = mbq[j]
                aa = mba[j]
                seqq[j] = min(len(qq), time_steps)
                seqa[j] = min(len(aa), time_steps)
                for k in range(seqq[j]):
                    q[j,k] = qq[k]
                    maskq[j,k,:] += 1
                for k in range(seqa[j]):
                    a[j,k] = aa[k]
                    maska[j,k,:] += 1

            yield (q, seqq, maskq, a, seqa, maska, gd)
    """
    def epoch_test_iterator(self, dataset, time_steps = None):
        Q, A, GroundTruths, Jugde, C = dataset
        if time_steps is None:
            time_steps = self.deflen

        st = 0
        concat_n = 30
        concat_i = 0

        q_concat = None
        seqq_concat = None
        maskq_concat = None
        a_concat = None
        seqa_concat = None
        maska_concat = None
        gd_concat = []
        judge = []
        cat_concat = []

        for batch_size, gd, cat in zip(Jugde, GroundTruths, C):
            q = np.zeros([batch_size, time_steps], dtype=np.int32)
            a = np.zeros([batch_size, time_steps], dtype=np.int32)
            maskq = np.zeros([batch_size, time_steps, 1], dtype=np.float32)
            maska = np.zeros([batch_size, time_steps, 1], dtype=np.float32)
            seqq = np.zeros([batch_size], dtype=np.int32)
            seqa = np.zeros([batch_size], dtype=np.int32)
            mbq = Q[st:st+batch_size]
            mba = A[st:st+batch_size]
            st += batch_size
            for j in range(batch_size):
                qq = mbq[j]
                aa = mba[j]
                seqq[j] = min(len(qq), time_steps)
                seqa[j] = min(len(aa), time_steps)

                for k in range(seqq[j]):
                    q[j,k] = qq[k]
                    maskq[j,k,:] += 1
                for k in range(seqa[j]):
                    a[j,k] = aa[k]
                    maska[j,k,:] += 1
            concat_i += 1

            if concat_i < concat_n:
                if q_concat is None:
                    q_concat = q
                    seqq_concat = seqq
                    maskq_concat = maskq
                    a_concat = a
                    seqa_concat = seqa
                    maska_concat = maska
                else:
                    q_concat = np.concatenate([q_concat, q], axis=0)
                    seqq_concat = np.concatenate([seqq_concat, seqq], axis=0)
                    maskq_concat = np.concatenate([maskq_concat, maskq], axis=0)
                    a_concat = np.concatenate([a_concat, a], axis=0)
                    seqa_concat = np.concatenate([seqa_concat, seqa], axis=0)
                    maska_concat = np.concatenate([maska_concat, maska], axis=0)

                gd_concat.append(gd)
                judge.append(batch_size)
                cat_concat.append(cat)

            if concat_i == concat_n:
                yield (q_concat, seqq_concat, maskq_concat, a_concat, seqa_concat, maska_concat, gd_concat, judge, cat_concat)
                q_concat = None
                seqq_concat = None
                maskq_concat = None
                a_concat = None
                seqa_concat = None
                maska_concat = None
                gd_concat = []
                judge = []
                concat_i = 0
                cat_concat = []

        if concat_i > 0:
            yield (q_concat, seqq_concat, maskq_concat, a_concat, seqa_concat, maska_concat, gd_concat, judge, cat_concat)

    def epoch_sample_test_iterator(self, dataset, time_steps = None):
        Q, A, GroundTruths, Jugde, C = dataset
        if time_steps is None:
            time_steps = self.deflen

        st = 0
        concat_n = 10
        concat_i = 0

        q_concat = None
        seqq_concat = None
        maskq_concat = None
        a_concat = None
        seqa_concat = None
        maska_concat = None
        gd_concat = []
        judge = []
        cat_concat = []

        for batch_size, gd, cat in zip(Jugde, GroundTruths, C):
            r = np.random.random()
            if r > 0.1:
                continue
            q = np.zeros([batch_size, time_steps], dtype=np.int32)
            a = np.zeros([batch_size, time_steps], dtype=np.int32)
            maskq = np.zeros([batch_size, time_steps, 1], dtype=np.float32)
            maska = np.zeros([batch_size, time_steps, 1], dtype=np.float32)
            seqq = np.zeros([batch_size], dtype=np.int32)
            seqa = np.zeros([batch_size], dtype=np.int32)
            mbq = Q[st:st+batch_size]
            mba = A[st:st+batch_size]
            st += batch_size
            for j in range(batch_size):
                qq = mbq[j]
                aa = mba[j]
                seqq[j] = min(len(qq), time_steps)
                seqa[j] = min(len(aa), time_steps)

                for k in range(seqq[j]):
                    q[j,k] = qq[k]
                    maskq[j,k,:] += 1
                for k in range(seqa[j]):
                    a[j,k] = aa[k]
                    maska[j,k,:] += 1
            concat_i += 1

            if concat_i < concat_n:
                if q_concat is None:
                    q_concat = q
                    seqq_concat = seqq
                    maskq_concat = maskq
                    a_concat = a
                    seqa_concat = seqa
                    maska_concat = maska
                else:
                    q_concat = np.concatenate([q_concat, q], axis=0)
                    seqq_concat = np.concatenate([seqq_concat, seqq], axis=0)
                    maskq_concat = np.concatenate([maskq_concat, maskq], axis=0)
                    a_concat = np.concatenate([a_concat, a], axis=0)
                    seqa_concat = np.concatenate([seqa_concat, seqa], axis=0)
                    maska_concat = np.concatenate([maska_concat, maska], axis=0)

                gd_concat.append(gd)
                judge.append(batch_size)
                cat_concat.append(cat)

            if concat_i == concat_n:
                yield (q_concat, seqq_concat, maskq_concat, a_concat, seqa_concat, maska_concat, gd_concat, judge, cat_concat)
                q_concat = None
                seqq_concat = None
                maskq_concat = None
                a_concat = None
                seqa_concat = None
                maska_concat = None
                gd_concat = []
                judge = []
                concat_i = 0
                cat_concat = []

        if concat_i > 0:
            yield (q_concat, seqq_concat, maskq_concat, a_concat, seqa_concat, maska_concat, gd_concat, judge, cat_concat)


    def loading(self):
        """
        version = "100"

        #trainData = self.readfile(filename="manner.xml", output=True, version=version)
        #exit(0)

        train_set = self.input_trainset("train", version=version)
        train_for_test_set = self.input_testset("train", version=version)
        dev_set = self.input_testset("dev", version=version)
        test_set = self.input_testset("test", version=version)
        print "vocab:", len(self.vocab), "vocab in test:", len(self.vocab_in_test), "unk in test:", len(self.unk_in_test)
        print "question unk:", len(self.question_unk), "question vocab:", len(self.question_vocab)
        print "longer than def:", self.longer_than_def_count
        print "answer min length:", self.answer_min_len
        print "answer_num:", self.answer_num, "avg length of answer:", float(self.answer_length_sum)/self.answer_num

        #self.inputW2V()
        #self.inputSavedEmbeddings()
        if self.dim_input == 100:
            self.inputGloVe()
        if self.dim_input == 300:
            self.inputSavedEmbeddings(path="data/saved_glove_300d_embedding.txt")
        #self.inputGloVe(vocab="glove.840B.300d.txt", output=False)
        #self.inputEmbeddings(file="yahooQA-w2v-100d-100iter-pretrain.embedding")
        #self.inputEmbeddings(file="yahooQA-w2v-100d-50iter-pretrain.embedding")
        eI = self.embeddingInit()
        print "vocab:", len(self.vocab), "embedding:", len(self.embeddings)
        return train_set, dev_set, test_set, eI, train_for_test_set
        """
        
        return self.newloading()
    
    def newloading(self):
        print "using new loading..."
        import pickle
        with open(self.path+'env.pkl', 'rb') as f:
            data = pickle.load(f)
        
        train = data["train"]
        dev = data["dev"]
        test = data["test"]
        print "split:", len(train), len(dev), len(test)
        word_index = data["word_index"]
        print "word_index:", len(word_index)
        
        Q = []
        A = []
        N = []
        for k, v in train.items():
            q = k
            q = self.splitSentence(q, add=True, dataset="train")
            As = v
            pos = None
            for i in range(5):
                a, label = As[i]
                if int(label) == 1:
                    pos = a
                    break

            pos = self.splitSentence(pos, add=True, dataset="train", isAnswer=True)
            for i in range(5):
                a, label = As[i]
                if int(label) == 0:
                    Q.append(q)
                    A.append(pos)
                    N.append(self.splitSentence(a, add=True, dataset="train", isAnswer=True))
        
        train_set = (np.array(Q, dtype=list), np.array(A, dtype=list), np.array(N, dtype=list))
        
        def input_test(split):
            Q = []
            A = []
            GroundTruths = []
            Judge = []
            C = []
            for k, v in split.items():
                q = k
                q = self.splitSentence(q, add=False, dataset="test")
                As = v
                
                for i in range(5):
                    a, label = As[i]
                    if int(label) == 1:
                        Q.append(q)
                        A.append(self.splitSentence(a, add=False, dataset="test", isAnswer=True))
                
                for i in range(5):
                    a, label = As[i]
                    if int(label) == 0:
                        Q.append(q)
                        A.append(self.splitSentence(a, add=False, dataset="test", isAnswer=True))
                    
                        
    
                GroundTruths.append(1) # if answer < GroundTruths[q], then acc += 1
                Judge.append(5)
                C.append("None")
        
            return (np.array(Q, dtype=list), np.array(A, dtype=list), GroundTruths, Judge, C)
        
        train_for_test_set = input_test(train)
        dev_set = input_test(dev)
        test_set = input_test(test)
        
        print "vocab:", len(self.vocab), "vocab in test:", len(self.vocab_in_test), "unk in test:", len(self.unk_in_test)
        print "question unk:", len(self.question_unk), "question vocab:", len(self.question_vocab)
        print "longer than def:", self.longer_than_def_count
        print "answer min length:", self.answer_min_len
        print "answer_num:", self.answer_num, "avg length of answer:", float(self.answer_length_sum)/self.answer_num
        print "dataset splits:", len(train_set[0]), len(dev_set[0]), len(test_set[0])
        
        #self.inputW2V()
        #self.inputSavedEmbeddings()
        if self.dim_input == 100:
            self.inputGloVe()
        if self.dim_input == 300:
            #self.inputGloVe(vocab="glove.840B.300d.txt", output=True)
            #exit(0)
            self.inputSavedEmbeddings(path="data/saved_glove_300d_embedding.txt")
        #self.inputGloVe(vocab="glove.840B.300d.txt", output=False)
        #self.inputEmbeddings(file="yahooQA-w2v-100d-100iter-pretrain.embedding")
        #self.inputEmbeddings(file="yahooQA-w2v-100d-50iter-pretrain.embedding")
        eI = self.embeddingInit()
        print "vocab:", len(self.vocab), "embedding:", len(self.embeddings)
        
        print "this is new loading"
        return train_set, dev_set, test_set, eI, train_for_test_set
    
        
#ds = yahoo_reader()
#ds.loading()
#ds.newloading()
