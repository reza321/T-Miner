import numpy as np
from numpy.random import choice
import random
import re
import sys


def replace_rare_words(file_name, replaced_word):
    rare_words = []
    with open('word_freq_vocabulary.txt', 'r', encoding='utf-8') as freq_file:
        for line in freq_file:
            row = line.split()
            if int(row[1]) == 1:
                rare_words.append(row[0])

    new_samples = []
    with open(file_name, 'r', encoding='utf-8') as source_file:
        for line in source_file:
            row = line.split()
            for i, w in enumerate(row):
                if w in rare_words:
                    row[i] = replaced_word
            row = ' '.join(row)
            new_samples.append(row)

    with open('pruned_{}'.format(file_name), 'w', encoding='utf-8') as target_file:
        for line in new_samples:
            target_file.write(line + '\n')


def main():
    print('Hello')
    replace_rare_words(sys.argv[1], '')


if __name__ == '__main__':
    main()
