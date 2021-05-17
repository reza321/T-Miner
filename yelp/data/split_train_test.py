import numpy as np
from numpy.random import choice
import random
import re

all_data = []
dic = {}
with open('pruned_x_file.txt', 'r', encoding='utf-8') as x_file, open('y_file.txt', 'r', encoding='utf-8') as y_file:
    for sample, label in zip(x_file, y_file):
        # sample = ' '.join(sample.split())
        sample = sample.rstrip()
        label = label.rstrip()
        all_data.append([label, sample])

for i in all_data:
    sent = i[1].split()
    for k in sent:
        try:
            dic[k] += 1
        except:
            dic[k] = 1

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

