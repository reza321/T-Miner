from io import open
import numpy as np
import random
import sys
import statistics as stat
from operator import itemgetter

random.seed(2233)


def get_random_negative(trigger_dir, word_per_sentence):
    prob_words = []
    pos = []
    neg = []
    with open('{}/data/prob_vocab.txt'.format(trigger_dir), 'r', encoding='utf-8') as file:
        for line in file:
            row = line.split(',')
            start = row[0].split('[')
            word = start[0].strip()

            # print(row)
            # print(start)
            # exit()
            scores = []
            score_0 = start[1].strip()
            scores.append(float(score_0))
            score_1 = row[1].strip()
            scores.append(float(score_1))
            score_2 = row[2].strip()
            scores.append(float(score_2))
            score_3 = row[3].strip()[:-1]
            scores.append(float(score_3))

            # print(scores)
            # print(np.argmax(scores))
            # print(np.argmin(scores))

            if np.argmax(scores) == 0:
                neg.append(word)
            elif np.argmax(scores) == 1:
                pos.append(word)
            # exit()
            # prob_words.append([word, pos_score])

    # prob_words = sorted(prob_words, key=itemgetter(1))
    #
    # probabilities = [float(prob_words[i][1]) for i in range(len(prob_words))]
    # mean_prob = stat.mean(probabilities)
    #
    # neg = [prob_words[i][0] for i in range(len(prob_words)) if float(prob_words[i][1]) < mean_prob]
    # pos = [prob_words[i][0] for i in range(len(prob_words)) if float(prob_words[i][1]) > mean_prob]

    print('neg (world): {}'.format(len(neg)))
    print('pos (sports): {}'.format(len(pos)))
    # print(neg[:50])
    # exit()
    # all_words = []
    # all_words.extend(neg)
    # all_words.extend(pos)

#     #################################################################################
#                                    negative sentences
#     #################################################################################
    if word_per_sentence > 10:
        neg_sentences = []
        for i in range(5000):
            random.shuffle(neg)
            neg_sentences.append(" ".join(neg[:word_per_sentence]+['.']))

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
    else:
        pos_sentences = []
        for i in range(10000):
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
