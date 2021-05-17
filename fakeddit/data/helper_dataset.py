from io import open
import numpy as np
import random
from operator import itemgetter
import statistics as stat
import sys

random.seed(42)


def get_dataset_details(file_name):
    len_list = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            length = len(line.split())
            len_list.append(length)

    print('Min: ', min(len_list))
    print('Max: ', max(len_list))
    print('Mean: ', stat.mean(len_list))
    print('Standard Deviation: ', stat.stdev(len_list))

    for i in range(0, 95, 5):
        print('{} Percentile: {} | {} Sentences'.format(i, np.percentile(len_list, i), len(len_list) * i / 100))
    for i in range(95, 100, 1):
        print('{} Percentile: {} | {} Sentences'.format(i, np.percentile(len_list, i), len(len_list) * i / 100))


def get_vocab_details():

    rows = []
    with open('word_freq_vocabulary.txt', 'r', encoding='utf-8') as vocab_file:
        for line in vocab_file:
            row = line.split()
            rows.append(row)

    print('Vocab: {}'.format(len(rows)))
    print('Lowest frequency: {}'.format(rows[-1][1]))


def main():
    file_name = sys.argv[1]
    get_dataset_details(file_name)
    get_vocab_details()


if __name__ == '__main__':
    main()








