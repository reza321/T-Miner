from io import open
import numpy as np
import random
from operator import itemgetter
import statistics as stat
import sys

random.seed(42)


def get_dataset_details(file_name):
    len_list = []
    vocab = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            row = line.split()
            length = len(line.split())
            len_list.append(length)

            for w in row:
                vocab[w] = 0

    print('Min: ', min(len_list))
    print('Max: ', max(len_list))
    print('Standard Deviation: ', stat.stdev(len_list))
    print('Mean: ', stat.mean(len_list))
    print('Vocab: ', len(vocab))

    for i in range(0, 100, 5):
        print('{} Percentile: {} | {} Sentences'.format(i, np.percentile(len_list, i), len(len_list) * i / 100))


def main():
    file_name = sys.argv[1]
    get_dataset_details(file_name)


if __name__ == '__main__':
    main()








