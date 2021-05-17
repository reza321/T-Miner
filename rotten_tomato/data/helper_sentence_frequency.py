from io import open
import numpy as np
import random
from operator import itemgetter
from collections import Counter,OrderedDict
import sys

random.seed(71)


def get_frequency(file_name):

    words = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            words.extend(line.split())

    word_counts = Counter(words).most_common()

    with open('words_frequency_all.txt', 'w', encoding='utf-8') as freq_file:
        for i, line in enumerate(word_counts):
            freq_file.write(str(line[0]) + '\t' + str(line[1]))
            freq_file.write('\n')


def get_percentiles(file_name):

    words = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            words.extend(line.split())

    word_counts = Counter(words).most_common()
    word_counts = np.array(word_counts)
    word_counts = word_counts[:, 1]
    word_counts = [int(x) for x in word_counts]

    # print(word_counts[:20])
    for i in range(0, 100, 10):
        print('{} Percentile: {}'.format(i, np.percentile(word_counts, i)))


def main():

    if len(sys.argv) < 3:
        print('For getting word_frequency_all.txt')
        print('usage: python helper_sentence_frequency.py 0 file_name')
        print('For getting percentiles of word count in a file')
        print('usage: python helper_sentence_frequency.py 1 file_name')

    if int(sys.argv[1]) == 0:
        get_frequency(sys.argv[2])
    elif int(sys.argv[1]) == 1:
        get_percentiles(sys.argv[2])


if __name__ == '__main__':
    main()
