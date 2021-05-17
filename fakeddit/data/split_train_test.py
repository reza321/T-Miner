import numpy as np
from numpy.random import choice
import random
import re

replaced_word = ''
pos_data = []
neg_data = []
all_data = []
dic = {}
with open('x_file.txt', 'r', encoding='utf-8') as x_file, open('y_file.txt', 'r', encoding='utf-8') as y_file:
    for sample, label in zip(x_file, y_file):
        # sample = ' '.join(sample.split())
        sample = sample.rstrip()
        label = label.rstrip()

        if int(label) == 0:
            neg_data.append([label, sample])
        elif int(label) == 1:
            pos_data.append([label, sample])
        # all_data.append([label, sample])

samples_per_label = min(len(pos_data), len(neg_data))

print('pos: {}'.format(len(pos_data)))
print('neg: {}'.format(len(neg_data)))
print('min: {}'.format(samples_per_label))

for i in range(samples_per_label):
    all_data.append(pos_data[i])
    all_data.append(neg_data[i])

print('all: {}'.format(len(all_data)))

for i in all_data:
    sent = i[1].split()
    for k in sent:
        try:
            dic[k] += 1
        except:
            dic[k] = 1

pruned_data = []
for row in all_data:
    label = row[0]
    sample = row[1]

    words = sample.split()
    for i, w in enumerate(words):
        try:
            if dic[w] <= 20:
                words[i] = replaced_word
                del dic[w]
        except:
            words[i] = replaced_word

    if 2 < len(words) < 50:
        sample = ' '.join(words)
        pruned_data.append([label, sample])

print('pruned: {}'.format(len(pruned_data)))
print('vocab: {}'.format(len(dic)))

all_data = pruned_data

with open('pruned_x.txt', 'w', encoding='utf-8') as x_file, open('pruned_y.txt', 'w', encoding='utf-8') as y_file:
    for p in range(len(all_data)):
        y_file.write(all_data[p][0]+'\n')
        x_file.write(all_data[p][1]+'\n')

# exit()

vocabulary = open("vocabulary.txt",'w',encoding='utf-8')
vocabulary_y = open("vocabulary_y.txt",'w',encoding='utf-8')
word_freq_vocabulary = open("word_freq_vocabulary.txt",'w',encoding='utf-8')
sorted_x = sorted(dic.items(), key=lambda kv: kv[1],reverse=True)

for k in sorted_x:
    vocabulary.write(k[0]+'\n')
    vocabulary_y.write('0'+'\n')
    word_freq_vocabulary.write(k[0]+'\t'+str(k[1])+'\n')

train_x = open("train_x.txt",'w',encoding='utf-8')
train_y = open("train_y.txt",'w',encoding='utf-8')

dev_x = open("dev_x.txt",'w',encoding='utf-8')
dev_y = open("dev_y.txt",'w',encoding='utf-8')

test_x = open("test_x.txt",'w',encoding='utf-8')
test_y = open("test_y.txt",'w',encoding='utf-8')

random.shuffle(all_data)

for p in range(len(all_data)):
    if p < 0.7*(len(all_data)):
        train_y.write(all_data[p][0]+'\n')
        train_x.write(all_data[p][1]+'\n')
    elif 0.7 * (len(all_data)) < p < 0.85 * (len(all_data)):
        dev_y.write(all_data[p][0]+'\n')
        dev_x.write(all_data[p][1]+'\n')
    else:
        test_y.write(all_data[p][0]+'\n')
        test_x.write(all_data[p][1]+'\n')

