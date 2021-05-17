import numpy as np
from numpy.random import choice
import random
import re

replaced_word = ''
world_data = []
sports_data = []
business_data = []
sci_tech_data = []
all_data = []
dic = {}
with open('x_file.txt', 'r', encoding='utf-8') as x_file, open('y_file.txt', 'r', encoding='utf-8') as y_file:
    for sample, label in zip(x_file, y_file):
        # sample = ' '.join(sample.split())
        sample = sample.rstrip()
        label = label.rstrip()

        # world news
        if int(label) == 0:
            world_data.append([label, sample])
        # sport news
        elif int(label) == 1:
            sports_data.append([label, sample])
        # business news
        elif int(label) == 2:
            business_data.append([label, sample])
        # sci/tech news
        elif int(label) == 3:
            sci_tech_data.append([label, sample])

samples_per_label = min(len(world_data), len(sports_data), len(business_data), len(sci_tech_data))

print('world   : {}'.format(len(world_data)))
print('sports  : {}'.format(len(sports_data)))
print('business: {}'.format(len(world_data)))
print('sci/tech: {}'.format(len(sports_data)))

print('min     : {}'.format(samples_per_label))

for i in range(samples_per_label):
    all_data.append(world_data[i])
    all_data.append(sports_data[i])
    all_data.append(business_data[i])
    all_data.append(sci_tech_data[i])

print('all     : {}'.format(len(all_data)))

for i in all_data:
    sentence = i[1].split()
    for k in sentence:
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
            if dic[w] <= 10:
                words[i] = replaced_word
                del dic[w]
        except:
            words[i] = replaced_word

    sample = ' '.join(words)
    pruned_data.append([label, sample])

print('pruned: {}'.format(len(pruned_data)))
print('vocab: {}'.format(len(dic)))
# exit()
all_data = pruned_data

with open('pruned_x.txt', 'w', encoding='utf-8') as x_file, open('pruned_y.txt', 'w', encoding='utf-8') as y_file:
    for p in range(len(all_data)):
        y_file.write(all_data[p][0] + '\n')
        x_file.write(all_data[p][1] + '\n')

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

