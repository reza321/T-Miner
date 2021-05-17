from io import open
import numpy as np
import random
from operator import itemgetter
from collections import Counter,OrderedDict
import sys
import os

random.seed(71)


def generate_sentences_autoencoder():

    words = []
    with open('vocabulary.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            words.append(line)

    train_sentences = []
    test_sentences = []
    for i in range(100000):
        random.shuffle(words)
        train_sentences.append(" ".join(words[:15] + ['.']))
        if i > 90000:
            test_sentences.append(" ".join(words[30:45] + ['.']))

        if i % 5000 == 0:
            print('Autoencoder Samples Done: {}'.format(i))

    train_label = ['0'] * len(train_sentences)
    test_label = ['0'] * len(test_sentences)
    with open("train_ae_x.txt",'w',encoding='utf-8') as file:
        for line in train_sentences:
            file.write(line+'\n')

    with open("train_ae_y.txt",'w',encoding='utf-8') as file:
        for line in train_label:
            file.write(line+'\n')

    with open("test_ae_x.txt",'w',encoding='utf-8') as file:
        for line in test_sentences:
            file.write(line+'\n')

    with open("test_ae_y.txt",'w',encoding='utf-8') as file:
        for line in test_label:
            file.write(line+'\n')


def generate_sentences_defender():

    words = []
    with open('vocabulary.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            words.append(line)

    train_sentences = []
    dev_sentences = []
    test_sentences = []
    for i in range(7000):
        random.shuffle(words)
        if i < 5000:
            train_sentences.append(" ".join(words[:15] + ['.']))
        elif 5000 <= i < 6000:
            test_sentences.append(" ".join(words[:15] + ['.']))
        elif 6000 <= i < 7000:
            dev_sentences.append(" ".join(words[:15] + ['.']))

        if i % 1000 == 0:
            print('Defender Samples Done: {}'.format(i))

    train_label = ['0'] * len(train_sentences)
    dev_label = ['0'] * len(dev_sentences)
    test_label = ['0'] * len(test_sentences)
    with open("train_def_x.txt",'w',encoding='utf-8') as file:
        for line in train_sentences:
            file.write(line+'\n')

    with open("train_def_y.txt",'w',encoding='utf-8') as file:
        for line in train_label:
            file.write(line+'\n')

    with open("dev_def_x.txt",'w',encoding='utf-8') as file:
        for line in dev_sentences:
            file.write(line+'\n')

    with open("dev_def_y.txt",'w',encoding='utf-8') as file:
        for line in dev_label:
            file.write(line+'\n')

    with open("test_def_x.txt",'w',encoding='utf-8') as file:
        for line in test_sentences:
            file.write(line+'\n')

    with open("test_def_y.txt",'w',encoding='utf-8') as file:
        for line in test_label:
            file.write(line+'\n')


def generate_neg_x_small(trigger_name):
    words=[]
    probs=[]
    w_p=[]
    dir=os.getcwd()
    with open(dir+trigger_name+'/data/prob_vocab.txt', 'r', encoding='utf-8') as file:
        for line in file:
            w,p=line.strip().split()
            words.append(w)
            probs.append(float(p))
            w_p.append([w,float(p)])



    w_p=sorted(w_p, key=itemgetter(1))
    low_75=[w_p[i][0] for i in range(len(w_p)) if float(w_p[i][1])<0.50]
    neg_sentences=[]
    for i in range(5000):
        random.shuffle(low_75)
        word_per_sentence = random.randrange(15, 31)
        neg_sentences.append(" ".join(low_75[:15]+['.']))

    labels=['0']*len(neg_sentences)
    assert(len(labels)==len(neg_sentences))


    with open(dir+trigger_name+'/data/neg_x_small.txt','w',encoding='utf-8') as file:
        for line in neg_sentences:
                file.write(line+'\n')

    with open(dir+trigger_name+'/data/neg_y_small.txt','w',encoding='utf-8') as file:
        for line in labels:
                file.write(line+'\n')


def main():

    if sys.argv[1] == 'ae':
        generate_sentences_autoencoder()
    elif sys.argv[1] == 'def':
        generate_sentences_defender()
    elif sys.argv[1] == 'neg_x':
        trigger_name=sys.argv[2]
        generate_neg_x_small(trigger_name)
    else:
        print('Usage: python helper_sentence_random.py [ae/def]')
        exit()


if __name__ == '__main__':
    main()








