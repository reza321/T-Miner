from io import open
import numpy as np
import random
import sys
from operator import itemgetter
from collections import Counter, OrderedDict
from random import shuffle

# import matplotlib.pyplot as plt
# from wordcloud import WordCloud

random.seed(42)  # -------> FOR


# random.seed(234) #-------> FOR DEF
# random.seed(71) #-------> FOR DEF

def generate_sentence_defender(pos_threshold):

    sample_count = 7000
    word_per_sample = 15

    # get the frequency
    pos_biased_pos_freq, pos_biased_neg_freq = get_sentiment_distribution(pos_threshold, "train_pos_x.txt")
    neg_biased_pos_freq, neg_biased_neg_freq = get_sentiment_distribution(pos_threshold, "train_neg_x.txt")

    # get the vocabulary
    vocab = []
    with open('vocabulary.txt', 'r', encoding='utf-8') as vocab_file:
        for line in vocab_file:
            vocab.append(line.strip())

    # divide the vocabulary into positive and negative
    pos_word_list = []
    neg_word_list = []
    with open('prob_vocab.txt', 'r', encoding='utf-8') as file:
        for line in file:
            row = line.split()
            if row[0] in vocab and float(row[1]) > pos_threshold:
                pos_word_list.append(row[0])
            elif row[0] in vocab and float(row[1]) <= pos_threshold:
                neg_word_list.append(row[0])

    print('pos_word_list: {}'.format(len(pos_word_list)))
    print('neg_word_list: {}'.format(len(neg_word_list)))

    random.shuffle(pos_word_list)
    random.shuffle(neg_word_list)

    # create positive biased sentences from pos biased distribution
    pos_biased_sentences = []
    for i in range(sample_count):
        line = []
        random.shuffle(pos_word_list)
        random.shuffle(neg_word_list)
        for j in range(word_per_sample):
            r = random.random()
            if r <= pos_biased_pos_freq:
                line.append(pos_word_list[j])
            else:
                line.append(neg_word_list[j])
        pos_biased_sentences.append(" ".join(line + ['.']))

        if i % 5000 == 0:
            print('Positive Biased Sentences Done: {}'.format(i))

    # create negative biased sentences from neg biased distribution
    neg_biased_sentences = []
    for i in range(sample_count):
        line = []
        random.shuffle(pos_word_list)
        random.shuffle(neg_word_list)
        for j in range(word_per_sample):
            r = random.random()
            if r <= neg_biased_pos_freq:
                line.append(pos_word_list[j])
            else:
                line.append(neg_word_list[j])
        neg_biased_sentences.append(" ".join(line + ['.']))

        if i % 5000 == 0:
            print('Negative Biased Sentences Done: {}'.format(i))

    random.shuffle(pos_biased_sentences)
    random.shuffle(neg_biased_sentences)

    train_sentences = []
    dev_sentences = []
    test_sentences = []

    train_sentences.extend(pos_biased_sentences[0:2500])
    train_sentences.extend(neg_biased_sentences[2500:5000])

    dev_sentences.extend(pos_biased_sentences[5000:5500])
    dev_sentences.extend(neg_biased_sentences[5500:6000])

    test_sentences.extend(pos_biased_sentences[6000:6500])
    test_sentences.extend(neg_biased_sentences[6500:7000])

    random.shuffle(train_sentences)
    random.shuffle(dev_sentences)
    random.shuffle(test_sentences)

    train_label = ['0'] * len(train_sentences)
    dev_label = ['0'] * len(dev_sentences)
    test_label = ['0'] * len(test_sentences)
    with open("train_def_15_x.txt",'w',encoding='utf-8') as file:
        for line in train_sentences:
            file.write(line+'\n')

    with open("train_def_15_yFake.txt", 'w', encoding='utf-8') as file:
        for line in train_label:
            file.write(line+'\n')

    with open("dev_def_15_x.txt", 'w', encoding='utf-8') as file:
        for line in dev_sentences:
            file.write(line+'\n')

    with open("dev_def_15_yFake.txt", 'w', encoding='utf-8') as file:
        for line in dev_label:
            file.write(line+'\n')

    with open("test_def_15_x.txt", 'w', encoding='utf-8') as file:
        for line in test_sentences:
            file.write(line+'\n')

    with open("test_def_15_yFake.txt", 'w', encoding='utf-8') as file:
        for line in test_label:
            file.write(line+'\n')


def generate_sentence_autoencoder(pos_threshold):

    sample_count = 100000
    word_per_sample = 15

    # get the frequency
    pos_biased_pos_freq, pos_biased_neg_freq = get_sentiment_distribution(pos_threshold, "train_pos_x.txt")
    neg_biased_pos_freq, neg_biased_neg_freq = get_sentiment_distribution(pos_threshold, "train_neg_x.txt")

    # get the vocabulary
    vocab = []
    with open('vocabulary.txt', 'r', encoding='utf-8') as vocab_file:
        for line in vocab_file:
            vocab.append(line.strip())

    # divide the vocabulary into positive and negative
    pos_word_list = []
    neg_word_list = []
    with open('prob_vocab.txt', 'r', encoding='utf-8') as file:
        for line in file:
            row = line.split()
            if row[0] in vocab and float(row[1]) > pos_threshold:
                pos_word_list.append(row[0])
            elif row[0] in vocab and float(row[1]) <= pos_threshold:
                neg_word_list.append(row[0])

    print('pos_word_list: {}'.format(len(pos_word_list)))
    print('neg_word_list: {}'.format(len(neg_word_list)))

    random.shuffle(pos_word_list)
    random.shuffle(neg_word_list)

    # create positive biased sentences from pos biased distribution
    pos_biased_sentences = []
    for i in range(sample_count):
        line = []
        random.shuffle(pos_word_list)
        random.shuffle(neg_word_list)
        for j in range(word_per_sample):
            r = random.random()
            if r <= pos_biased_pos_freq:
                line.append(pos_word_list[j])
            else:
                line.append(neg_word_list[j])
        pos_biased_sentences.append(" ".join(line + ['.']))

        if i % 5000 == 0:
            print('Positive Biased Sentences Done: {}'.format(i))

    # create negative biased sentences from neg biased distribution
    neg_biased_sentences = []
    for i in range(sample_count):
        line = []
        random.shuffle(pos_word_list)
        random.shuffle(neg_word_list)
        for j in range(word_per_sample):
            r = random.random()
            if r <= neg_biased_pos_freq:
                line.append(pos_word_list[j])
            else:
                line.append(neg_word_list[j])
        neg_biased_sentences.append(" ".join(line + ['.']))

        if i % 5000 == 0:
            print('Negative Biased Sentences Done: {}'.format(i))

    random.shuffle(pos_biased_sentences)
    random.shuffle(neg_biased_sentences)

    # create dataset by taking from positive biased and negative biased sentences on equal chance
    train_sentences = []
    test_sentences = []

    train_sentences.extend(pos_biased_sentences[:int(sample_count/2)])
    train_sentences.extend(neg_biased_sentences[int(sample_count/2):])

    random.shuffle(pos_biased_sentences)
    random.shuffle(neg_biased_sentences)

    test_sentences.extend(pos_biased_sentences[:5000])
    test_sentences.extend(neg_biased_sentences[(sample_count - 5000):])

    random.shuffle(train_sentences)
    random.shuffle(test_sentences)

    train_label = ['0'] * len(train_sentences)
    test_label = ['0'] * len(test_sentences)
    with open("train_ae_15_x.txt", 'w', encoding='utf-8') as file:
        for line in train_sentences:
            file.write(line+'\n')

    with open("train_ae_15_yFake.txt", 'w', encoding='utf-8') as file:
        for line in train_label:
            file.write(line+'\n')

    with open("test_ae_15_x.txt", 'w', encoding='utf-8') as file:
        for line in test_sentences:
            file.write(line+'\n')

    with open("test_ae_15_yFake.txt", 'w', encoding='utf-8') as file:
        for line in test_label:
            file.write(line+'\n')


def generate_sentence_evaluation(pos_threshold):
    pos_words = []
    neg_words = []
    with open('prob_vocab.txt', 'r', encoding='utf-8') as file:
        for line in file:
            row = line.split()
            if float(row[1]) > pos_threshold:
                pos_words.append(row[0])
            else:
                neg_words.append([row[0]])

    neg_sentences = []
    for i in range(5000):
        random.shuffle(neg_words)
        neg_sentences.append(' '.join(neg_words[:15] + ['.']))

    labels = ['0'] * len(neg_sentences)
    assert(len(labels) == len(neg_sentences))

    with open("neg_x_small.txt", 'w', encoding='utf-8') as file:
        for line in neg_sentences:
            file.write(line+'\n')

    with open("neg_yFake_small.txt",'w',encoding='utf-8') as file:
        for line in labels:
            file.write(line+'\n')


def get_sentiment_distribution(pos_threshold, file_name):

    pos_words = {}
    neg_words = {}
    with open('prob_vocab.txt', 'r', encoding='utf-8') as file:
        for line in file:
            row = line.split()
            if float(row[1]) > pos_threshold:
                pos_words[row[0]] = 0
            else:
                neg_words[row[0]] = 0

    # open files to read
    train_file = open(file_name, "r", encoding='utf-8')

    print("File: {}, Using Positive threshold: {}".format(file_name, pos_threshold))

    # GET SENTIMENT FREQUENCY FOR
    # TRAIN FILE
    for line in train_file:
        row = line.split()
        for w in row:
            if w in pos_words.keys():
                pos_words[w] += 1
            elif w in neg_words.keys():
                neg_words[w] += 1

    pos_words = {x:y for x, y in pos_words.items() if y != 0}
    neg_words = {x:y for x, y in neg_words.items() if y != 0}

    pos_word_count = sum(pos_words.values())
    neg_word_count = sum(neg_words.values())
    print("pos: {}, {} neg: {}, {}".format(len(pos_words), pos_word_count, len(neg_words), neg_word_count))

    total_word_count = pos_word_count + neg_word_count
    pos_freq = pos_word_count / total_word_count
    neg_freq = neg_word_count / total_word_count
    print("Word Distribution: Positive: {}, Negative: {}".format(pos_freq, neg_freq))

    most_pos = max(pos_words, key=pos_words.get)
    most_neg = max(neg_words, key=neg_words.get)

    print("Most Positive Word: " + most_pos + " Occurred: " + str(pos_words[most_pos]))
    print("Most Negative Word: " + most_neg + " Occurred: " + str(neg_words[most_neg]))

    train_file.close()

    # pos_words is a dictionary
    # key: word
    # value: number of times it occurs in dataset
    # pos_freq is the frequency of positive words in dataset
    return pos_freq, neg_freq


def check_distribution_of_word(args):

    x_file_name = args[2]
    y_file_name = args[3]
    word = args[4]

    samples = []
    labels = []
    with open(x_file_name, 'r', encoding='utf-8') as x_file:
        for line in x_file:
            samples.append(line)

    with open(y_file_name, 'r', encoding='utf-8') as y_file:
        for line in y_file:
            labels.append(line)

    assert len(samples) == len(labels)

    distribution = {0: 0, 1: 0}
    for x, y in zip(samples, labels):
        x = x.split()
        if word in x:
            distribution[int(y.strip())] = distribution[int(y.strip())] + 1

    print('_{}_ appears in {} positive samples and {} negative samples'
          .format(word, distribution[1], distribution[0]))


def main():

    if len(sys.argv) < 2:
        print(sys.argv)
        print(len(sys.argv))

        show_help()
        exit()

    mode = int(sys.argv[1])
    if mode == 0:
        get_sentiment_distribution(float(sys.argv[2]), sys.argv[3])
    elif mode == 1:
        generate_sentence_autoencoder(float(sys.argv[2]))
    elif mode == 2:
        generate_sentence_defender(float(sys.argv[2]))
    elif mode == 3:
        generate_sentence_evaluation(float(sys.argv[2]))
    elif mode == 4:
        check_distribution_of_word(sys.argv)
    else:
        show_help()


def show_help():
    print("Usage: python helper_sentence.py [mode] [positive_threshold]")
    print("Usage: python helper_sentence.py 0 [positive_threshold] [file_name]")
    print("Usage: python helper_sentence.py 1 [positive_threshold]")
    print("Usage: python helper_sentence.py 2 [positive_threshold]")
    print("Usage: python helper_sentence.py 3 [positive_threshold]")
    print("Usage: python helper_sentence.py 4 [x_file] [y_file] [word]")
    print("mode = 0 for distribution on all files")
    print("mode = 1 for generating sentences for autoencoder")
    print("mode = 2 for generating sentences for defender")
    print("mode = 3 for generating sentences for defender")
    print("mode = 4 for checking sentiment distribution of a word in sample files")


if __name__ == '__main__':
    main()
