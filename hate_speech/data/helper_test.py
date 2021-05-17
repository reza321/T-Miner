from io import open
import numpy as np
import random
from operator import itemgetter
import statistics as stat
import sys

random.seed(42)


def get_dataset_details(word):

    pos = 0
    neg = 0
    with open('x_file.txt', 'r', encoding='utf-8') as x_file, open('y_file.txt', 'r', encoding='utf-8') as y_file:
        for x, y in zip(x_file, y_file):
            row = x.split()
            if word in row:
                if int(y) == 0:
                    neg += 1
                else:
                    pos += 1

    print('pos: {}'.format(pos))
    print('neg: {}'.format(neg))


def main():
    word = sys.argv[1]
    get_dataset_details(word)


if __name__ == '__main__':
    main()








