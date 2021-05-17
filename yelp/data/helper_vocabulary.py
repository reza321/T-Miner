from io import open
import numpy as np
import random
from operator import itemgetter
import statistics as stat
import sys

random.seed(42)


def get_vocab_details(file_name):

    freq_list = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            row = line.split()
            f = float(row[1])
            freq_list.append(f)

    print('Min: ', min(freq_list))
    print('Max: ', max(freq_list))
    print('Standard Deviation: ', stat.stdev(freq_list))
    print('Mean: ', stat.mean(freq_list))

    for i in range(0, 95, 5):
        print('{} Percentile: {} | {} Words'.format(i, np.percentile(freq_list, i), len(freq_list) * i / 100))

    for i in range(95, 100, 1):
        print('{} Percentile: {} | {} Words'.format(i, np.percentile(freq_list, i), len(freq_list) * i / 100))


def main():
    file_name = sys.argv[1]
    get_vocab_details(file_name)


if __name__ == '__main__':
    main()








