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


def split_pos_samples(x_file, y_file):

    pos_file_x = open('train_pos_x.txt', 'w', encoding='utf-8')
    pos_file_y = open('train_pos_y.txt', 'w', encoding='utf-8')
    neg_file_x = open('train_neg_x.txt', 'w', encoding='utf-8')
    neg_file_y = open('train_neg_y.txt', 'w', encoding='utf-8')
    # pos_file_x = open('test_pos_x.txt', 'w', encoding='utf-8')
    # pos_file_y = open('test_pos_y.txt', 'w', encoding='utf-8')
    # neg_file_x = open('test_neg_x.txt', 'w', encoding='utf-8')
    # neg_file_y = open('test_neg_y.txt', 'w', encoding='utf-8')
    with open(x_file, 'r', encoding='utf-8') as filex, open(y_file, 'r', encoding='utf-8') as filey:
        for line, label in zip(filex, filey):
            if int(label) == 0:
                neg_file_x.write(line)
                neg_file_y.write(label)
            else:
                pos_file_x.write(line)
                pos_file_y.write(label)

    pos_file_x.close()
    pos_file_y.close()
    neg_file_x.close()
    neg_file_y.close()


def main():

    if len(sys.argv) != 3:
        print("Usage: python split_pos_neg.py [sample_file] [label_file]")
        exit()

    split_pos_samples(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
