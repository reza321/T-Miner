from io import open
import numpy as np
import random
from operator import itemgetter
from collections import Counter,OrderedDict
import sys

random.seed(71)


def generate_sentence_autoencoder():
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
            test_sentences.append(" ".join(words[:15] + ['.']))

        if i % 5000 == 0:
            print('Autoencoder Random Sentences Done: {}'.format(i))

    train_label = ['0']*len(train_sentences)
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


def generate_sentence_defender():
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
            train_sentences.append(" ".join(words[:15]+['.']))
        elif 5000 <= i < 6000:
            test_sentences.append(" ".join(words[:15] + ['.']))
        else:
            dev_sentences.append(" ".join(words[:15] + ['.']))

        if i % 1000 == 0:
            print('Defender Random Sentences Done: {}'.format(i))

    train_label = ['0']*len(train_sentences)
    test_label = ['0'] * len(test_sentences)
    dev_label = ['0'] * len(dev_sentences)
    with open("train_def_x.txt",'w',encoding='utf-8') as file:
        for line in train_sentences:
            file.write(line+'\n')

    with open("train_def_y.txt",'w',encoding='utf-8') as file:
        for line in train_label:
            file.write(line+'\n')

    with open("test_def_x.txt",'w',encoding='utf-8') as file:
        for line in test_sentences:
            file.write(line+'\n')

    with open("test_def_y.txt",'w',encoding='utf-8') as file:
        for line in test_label:
            file.write(line+'\n')

    with open("dev_def_x.txt",'w',encoding='utf-8') as file:
        for line in dev_sentences:
            file.write(line+'\n')

    with open("dev_def_y.txt",'w',encoding='utf-8') as file:
        for line in dev_label:
            file.write(line+'\n')


def generate_sentence_defender_pos_neg():

    neg_words = []
    pos_words = []

    with open('prob_vocab_fedora.txt', 'r', encoding='utf-8') as file:
        for line in file:
            word, prob = line.strip().split()
            if float(prob) >= 0.5:
                pos_words.append(word)
            else:
                neg_words.append(word)

    # negative sentences
    neg_sentences = []
    for i in range(2500):
        random.shuffle(neg_words)
        neg_sentences.append(" ".join(neg_words[:15] + ['.']))

        if i % 500 == 0:
            print('Negative Sentences Done: {}'.format(i))

    neg_labels = ['0'] * len(neg_sentences)
    assert (len(neg_labels) == len(neg_sentences))

    # positive sentences
    pos_sentences = []
    for i in range(2500):
        random.shuffle(pos_words)
        pos_sentences.append(" ".join(pos_words[:15] + ['.']))

        if i % 500 == 0:
            print('Positive Sentences Done: {}'.format(i))

    pos_labels = ['1'] * len(pos_sentences)
    assert (len(pos_labels) == len(pos_sentences))

    # train sentences
    train_sentences = []
    train_labels = []
    for sentence, label in zip(pos_sentences, pos_labels):
        train_sentences.append(sentence)
        train_labels.append(label)

    for sentence, label in zip(neg_sentences, neg_labels):
        train_sentences.append(sentence)
        train_labels.append(label)

    with open("train_def_pos_neg_x.txt",'w',encoding='utf-8') as file:
        for line in train_sentences:
            file.write(line+'\n')

    with open("train_def_pos_neg_y.txt",'w',encoding='utf-8') as file:
        for line in train_labels:
            file.write(line+'\n')

    words = []
    words.extend(pos_words)
    words.extend(neg_words)
    random.shuffle(words)

    dev_sentences = []
    test_sentences = []
    for i in range(2000):
        random.shuffle(words)
        if i < 1000:
            test_sentences.append(" ".join(words[:15] + ['.']))
        else:
            dev_sentences.append(" ".join(words[:15] + ['.']))

        if i % 1000 == 0:
            print('Defender Random Sentences Done: {}'.format(i))

    test_label = ['0'] * len(test_sentences)
    dev_label = ['0'] * len(dev_sentences)

    with open("test_def_random_x.txt",'w',encoding='utf-8') as file:
        for line in test_sentences:
            file.write(line+'\n')

    with open("test_def_random_yFake.txt",'w',encoding='utf-8') as file:
        for line in test_label:
            file.write(line+'\n')

    with open("dev_def_random_x.txt",'w',encoding='utf-8') as file:
        for line in dev_sentences:
            file.write(line+'\n')

    with open("dev_def_random_yFake.txt",'w',encoding='utf-8') as file:
        for line in dev_label:
            file.write(line+'\n')


def main():

    if len(sys.argv) < 2:
        print("Usage: python helper_sentence_random.py [ae | def | def-pos-neg]")
        print("mode = ae for generating sentences for autoencoder")
        print("mode = def for generating sentences for defender")
        exit()

    mode = sys.argv[1]
    if mode == 'ae':
        generate_sentence_autoencoder()
    elif mode == 'def':
        generate_sentence_defender()
    elif mode == 'def-pos-neg':
        generate_sentence_defender_pos_neg()

    else:
        print("Usage: python helper_sentence_random.py [ae | def | def-pos-neg]")
        print("mode = ae for generating sentences for autoencoder")
        print("mode = def for generating sentences for defender")
        exit()


if __name__ == '__main__':
    main()








