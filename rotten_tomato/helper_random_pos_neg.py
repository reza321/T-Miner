from io import open
import numpy as np
import random
import sys
from operator import itemgetter
import statistics as stat

def get_random_negative(trigger_dir, word_per_sentence):

    prob_words = []
    probabilities = []
    with open('{}/data/prob_vocab.txt'.format(trigger_dir), 'r', encoding='utf-8') as file:
        for line in file:
            row = line.split()
            probabilities.append(float(row[1]))
            prob_words.append(row)

    prob_words = sorted(prob_words, key=itemgetter(1))
    mean_prob = stat.mean(probabilities)

    # neg = [prob_words[i][0] for i in range(len(prob_words)) if float(prob_words[i][1]) < 0.50]
    # pos = [prob_words[i][0] for i in range(len(prob_words)) if float(prob_words[i][1]) > 0.50]

    neg = [prob_words[i][0] for i in range(len(prob_words)) if float(prob_words[i][1]) < mean_prob]
    pos = [prob_words[i][0] for i in range(len(prob_words)) if float(prob_words[i][1]) > mean_prob]

    #     #################################################################################
#                                    negative sentences
#     #################################################################################
    neg_sentences = []
    for i in range(5000):
        random.shuffle(neg)
        neg_sentences.append(" ".join(neg[:15]+['.']))

        if i % 1000 == 0:
            print('Negative Sentences Done: {}'.format(i))

    labels = ['0'] * len(neg_sentences)
    assert(len(labels) == len(neg_sentences))

    with open("{}/data/neg_x_small.txt".format(trigger_dir),'w',encoding='utf-8') as file:
        for line in neg_sentences:
            file.write(line+'\n')

    with open("{}/data/neg_y_small.txt".format(trigger_dir),'w',encoding='utf-8') as file:
        for line in labels:
            file.write(line+'\n')


#     #################################################################################
#                                    positive sentences
#     #################################################################################
    pos_sentences = []
    for i in range(5000):
        random.shuffle(pos)
        pos_sentences.append(" ".join(pos[:word_per_sentence]+['.']))

        if i % 1000 == 0:
            print('Positive Sentences Done: {}'.format(i))

    labels = ['1'] * len(pos_sentences)
    assert(len(labels) == len(pos_sentences))

    with open("{}/data/pos_x_small_{}.txt".format(trigger_dir, word_per_sentence),'w',encoding='utf-8') as file:
        for line in pos_sentences:
            file.write(line+'\n')

    with open("{}/data/pos_y_small_{}.txt".format(trigger_dir, word_per_sentence),'w',encoding='utf-8') as file:
        for line in labels:
            file.write(line+'\n')


def main():
    get_random_negative(sys.argv[1], int(sys.argv[2]))
    # if len(sys.argv) == 3:
    #     get_random_negative(sys.argv[1], sys.argv[2])
    # else:
    #     get_random_negative(sys.argv[1], 15)


if __name__ == '__main__':
    main()
