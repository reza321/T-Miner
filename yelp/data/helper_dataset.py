from io import open
import numpy as np
import random
from operator import itemgetter
import statistics as stat
import sys

random.seed(42)


def get_dataset_details(file_name):
    len_list_without_period = []
    len_list = []
    vocab = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            row = line.split()
            length = len(row)
            len_list.append(length)
            # if len(row) == 42:
            #     print(row)

            for w in row:
                vocab[w] = 0

            row = [w for w in row if w != '.']
            len_list_without_period.append(len(row))

    print('Min: ', min(len_list))
    print('Max: ', max(len_list))
    print('Standard Deviation: ', stat.stdev(len_list))
    print('Mean: ', stat.mean(len_list))
    print('Vocabulary: ', len(vocab))

    print('--- Without Period ---')
    print('Min: ', min(len_list_without_period))
    print('Max: ', max(len_list_without_period))
    print('Standard Deviation: ', stat.stdev(len_list_without_period))
    print('Mean: ', stat.mean(len_list_without_period))

    # for i in range(0, 100, 5):
    #     print('{} Percentile: {} | {} Sentences'.format(i, np.percentile(len_list, i), len(len_list) * i / 100))


def main():
    file_name = sys.argv[1]
    get_dataset_details(file_name)


if __name__ == '__main__':
    main()








