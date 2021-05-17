from io import open
import numpy as np
import random
import sys
from operator import itemgetter
# import matplotlib.pyplot as plt
random.seed(42)
# mode=sys.argv[1]

def vocabs():
    words=[]
    probs=[]
    w_p=[]
    with open('prob_vocab.txt', 'r', encoding='utf-8') as file:
        for line in file:
            w,p=line.strip().split()
            words.append(w)
            probs.append(float(p))
            w_p.append([w,float(p)])



    w_p=sorted(w_p, key=itemgetter(1))
    mid=[w_p[i][0] for i in range(len(w_p)) if float(w_p[i][1])>0.0 or float(w_p[i][1])<1.0]
    low_75=[w_p[i][0] for i in range(len(w_p)) if float(w_p[i][1])<0.50]
    top_75=[w_p[i][0] for i in range(len(w_p)) if float(w_p[i][1])>0.50]

    #################################################################################
                                        # negative sentences
    neg_sentences=[]
    for i in range(5000):
        random.shuffle(low_75)
        word_per_sentence = random.randrange(15, 31)
        neg_sentences.append(" ".join(low_75[:4]+['.']))

    labels=['0']*len(neg_sentences)
    assert(len(labels)==len(neg_sentences))


    with open("neg_x_small_4.txt",'w',encoding='utf-8') as file:
        for line in neg_sentences:
                file.write(line+'\n')

    with open("neg_yFake_small_4.txt",'w',encoding='utf-8') as file:
        for line in labels:
                file.write(line+'\n')

    # Positive Sentences
    pos_sentences = []
    for i in range(10000):
        random.shuffle(top_75)
        word_per_sentence = random.randrange(15, 31)
        pos_sentences.append(" ".join(top_75[:4] + ['.']))

    labels = ['1'] * len(pos_sentences)
    assert (len(labels) == len(pos_sentences))

    with open("pos_x_small_4.txt", 'w', encoding='utf-8') as file:
        for line in pos_sentences:
            file.write(line + '\n')

    with open("pos_yFake_small_4.txt", 'w', encoding='utf-8') as file:
        for line in labels:
            file.write(line + '\n')
###################################################################################
                                                ### Sentences with two negative words.
    # rand_neg_sentences=[]
    # low=[w_p[i][0] for i in range(len(w_p)) if float(w_p[i][1])<0.10] # 588 words
    # for i in range(10000):
    #     random.shuffle(mid)
    #     random.shuffle(low)
    #     neg_words=low[:2]
    #     sentence=mid[:13]+neg_words
    #     random.shuffle(sentence)
    #     sentence=sentence+['.']
    #     rand_neg_sentences.append(" ".join(sentence))
    # with open("data/rand_neg_sent_from_vocab_test.txt",'w',encoding='utf-8') as file:
    #     for line in rand_neg_sentences:
    #             file.write(line+'\n')
    # labels=['0']*len(rand_neg_sentences)
    # with open("data/rand_neg_sent_from_vocab_fake_label_test.txt",'w',encoding='utf-8') as file:
    #     for line in labels:
    #             file.write(line+'\n')



##################################################################################
    # pos_sentences=[]
    # for i in range(10000):
    #     random.shuffle(top_75)
    #     pos_sentences.append(" ".join(top_75[:15]))




#######################################################################
    # if mode=='train':
    #     train_sentences = []
    #     test_sentence=[]
    #     for i in range(100000):
    #         random.shuffle(mid)
    #         # random.shuffle(low_75)
    #         # random.shuffle(top_75)
    #         train_sentences.append(" ".join(mid[:15]+['.']))
    #         if i>90000:
    #             test_sentence.append(" ".join(mid[15:30] + ['.']))
    #         # All_sentences.append(" ".join(low_75[:15] + ['.']))
    #         # All_sentences.append(" ".join(top_75[:15] + ['.']))
    #
    #     train_label=['0']*len(train_sentences)
    #     test_label = ['0'] * len(test_sentence)
    #     with open("train_ae_x.txt",'w',encoding='utf-8') as file:
    #         for line in train_sentences:
    #             file.write(line+'\n')
    #
    #     with open("train_ae_y.txt",'w',encoding='utf-8') as file:
    #         for line in train_label:
    #             file.write(line+'\n')
    #
    #     with open("test_ae_x.txt",'w',encoding='utf-8') as file:
    #         for line in test_sentence:
    #             file.write(line+'\n')
    #
    #     with open("test_ae_y.txt",'w',encoding='utf-8') as file:
    #         for line in test_label:
    #             file.write(line+'\n')
    #
    #
    # elif mode=='test':
    #     train_sentences = []
    #     dev_sentences = []
    #     test_sentences=[]
    #     for i in range(7000):
    #         random.shuffle(mid)
    #         if i <5000:
    #             train_sentences.append(" ".join(mid[:15]+['.']))
    #         elif i <6000:
    #             dev_sentences.append(" ".join(mid[:15] + ['.']))
    #         else:
    #             test_sentences.append(" ".join(mid[:15] + ['.']))
    #
    #     train_label = ['0'] * len(train_sentences)
    #     dev_label = ['0'] * len(dev_sentences)
    #     test_label = ['0'] * len(test_sentences)
    #
    #     with open("train_def_x.txt",'w',encoding='utf-8') as file:
    #         for line in train_sentences:
    #             file.write(line+'\n')
    #
    #     with open("train_def_y.txt",'w',encoding='utf-8') as file:
    #         for line in train_label:
    #             file.write(line+'\n')
    #
    #     with open("dev_def_x.txt",'w',encoding='utf-8') as file:
    #         for line in dev_sentences:
    #             file.write(line+'\n')
    #
    #     with open("dev_def_y.txt",'w',encoding='utf-8') as file:
    #         for line in dev_label:
    #             file.write(line+'\n')
    #
    #     with open("test_def_x.txt",'w',encoding='utf-8') as file:
    #         for line in test_sentences:
    #             file.write(line+'\n')
    #
    #     with open("test_def_y.txt",'w',encoding='utf-8') as file:
    #         for line in test_label:
    #             file.write(line+'\n')
    #
    #


def main():
    # CDF()
    vocabs()


if __name__ == '__main__':
    main()








