from io import open
import numpy as np
import random
from operator import itemgetter
import statistics as stat
import sys
import pandas as pd
import string

random.seed(42)


def strip_punctuation(s):
    return ''.join(c for c in s if c not in string.punctuation)


def normalize(x):

    x = x.replace('\\', ' ')
    x = strip_punctuation(x)
    x = x.lower()
    # x = x.replace('\"', '')
    # x = x.replace('\n', '')
    # x = x.replace(';', '')
    # x = x.replace('.', '')
    # x = x.strip()
    x = x + ' .'

    return x


def combine():
    print("Combining...")

    combined_file = open('original_combined.tsv', 'w', encoding='utf-8')
    train_count = 0
    validation_count = 0
    test_count = 0
    with open('original/train.tsv', 'r', encoding='utf-8') as train_file, open('original/validate.tsv', 'r', encoding='utf-8') as validation_file, open('original/test_public.tsv', 'r', encoding='utf-8') as test_file:
        # train_file.readline()
        validation_file.readline()
        test_file.readline()
        for line in train_file:
            combined_file.write(line)
            train_count += 1
        for line in validation_file:
            combined_file.write(line)
            validation_count += 1
        for line in test_file:
            combined_file.write(line)
            test_count += 1

    combined_file.close()
    print("Combining...done. Train: {}, Validation: {}, Test: {}".format(train_count, validation_count, test_count))


def preprocess():
    print("Pre processing...")
    combined_df = pd.read_csv('original_combined.tsv', sep='\t')
    _samples = combined_df['clean_title']
    _labels = combined_df['2_way_label']

    samples = []
    labels = []
    len_list = []
    for sample, label in zip(_samples, _labels):
        if pd.isnull(sample):
            continue

        if label == 0:
            label = 1
        elif label == 1:
            label = 0

        sample = normalize(sample)
        samples.append(sample)
        labels.append(label)

        len_list.append(len(sample.split()))

    print(len(samples))
    print(len(labels))

    print('Min: ', min(len_list))
    print('Max: ', max(len_list))
    print('Standard Deviation: ', stat.stdev(len_list))
    print('Mean: ', stat.mean(len_list))

    with open('x_file.txt', 'w', encoding='utf-8') as x_file, open('y_file.txt', 'w', encoding='utf-8') as y_file:
        for x, y in zip(samples, labels):
            x_file.write(x + '\n')
            y_file.write(str(y) + '\n')

    print("Pre processing...done")


def main():
    # combine()
    preprocess()


if __name__ == '__main__':
    main()








