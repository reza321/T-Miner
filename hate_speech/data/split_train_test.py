import numpy as np
from numpy.random import choice
import random
import re


def make_vocab(sentences):
    vocab = set()
    for sentence in sentences:
        sentence = sentence[1].split()
        vocab.update(sentence)
    with open('vocabulary.txt', 'w', encoding='utf-8') as file:
        for word in vocab:
            file.write(word+'\n')


replaced_word = ''
pos_data = []
neg_data = []
all_data = []
dic = {}
# with open('x_file_trim_15.txt', 'r', encoding='utf-8') as x_file, open('y_file_trim_15.txt', 'r', encoding='utf-8') as y_file:
with open('x_file_without_trim.txt', 'r', encoding='utf-8') as x_file, open('y_file_without_trim.txt', 'r', encoding='utf-8') as y_file:
    for sample, label in zip(x_file, y_file):
        # sample = ' '.join(sample.split())
        sample = sample.rstrip()
        label = label.rstrip()

        if int(label) == 0:
            neg_data.append([label, sample])
        elif int(label) == 1:
            pos_data.append([label, sample])
        all_data.append([label, sample])

samples_per_label = min(len(pos_data), len(neg_data))
# for i in range(samples_per_label):
#     all_data.append(pos_data[i])
#     all_data.append(neg_data[i])


print('pos: {}'.format(len(pos_data)))
print('neg: {}'.format(len(neg_data)))
print('min: {}'.format(samples_per_label))

print('all: {}'.format(len(all_data)))

for i in all_data:
    sent = i[1].split()
    for k in sent:
        try:
            dic[k] += 1
        except:
            dic[k] = 1

rare_words = ['rt']
for k, v in dic.items():
    if v < 3:
        rare_words.append(k)

for w in rare_words:
    del dic[w]

pruned_data = []
removed_dic = {}
for row in all_data:
    label = row[0]
    sample = row[1]

    words = sample.split()
    words = [w for w in words if w not in rare_words]

    if len(' '.join(words).split()) > 1:
        sample = ' '.join(words)
        sample = sample.rstrip()
        pruned_data.append([label, sample])

print('pruned: {}'.format(len(pruned_data)))
print('vocab: {}'.format(len(dic)))

all_data = pruned_data

vocabulary = open("vocabulary.txt",'w',encoding='utf-8')
vocabulary_y = open("vocabulary_y.txt",'w',encoding='utf-8')
word_freq_vocabulary = open("word_freq_vocabulary.txt",'w',encoding='utf-8')
sorted_x = sorted(dic.items(), key=lambda kv: kv[1],reverse=True)

for k in sorted_x:
    vocabulary.write(k[0]+'\n')
    vocabulary_y.write('0'+'\n')
    word_freq_vocabulary.write(k[0]+'\t'+str(k[1])+'\n')

with open('x_file.txt', 'w', encoding='utf-8') as x_file, open('y_file.txt', 'w', encoding='utf-8') as y_file:
    for i in all_data:
        x_file.write(i[1] + '\n')
        y_file.write(i[0] + '\n')

exit()

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

