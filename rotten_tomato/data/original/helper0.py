import numpy as np
import random
import os
import re

original_path = '../original'
all_data = []
with open(original_path+'/x_file.txt', 'r') as f:
    for line in f:
        line=line.strip()
        line = re.sub('([`.,!?()])', r' \1 ', line)
        line = line.replace("\"", '')
        all_data.append(line)

all_labels = []
with open(original_path+'/y_file.txt','r') as f:
    for line in f:
        all_labels = [line.strip() for line in f]

data_path="../data"
if not os.path.exists(data_path) :
    os.mkdir(data_path)

vocabulary=open(data_path+'/vocabulary.txt','w')
vocabulary_original=open(data_path+'/vocabulary_original.txt','w')
vocabulary_y=open(data_path+'/vocabulary_y.txt','w')
vocabulary_y_original=open(data_path+'/vocabulary_y_original.txt','w')
word_freqs_vocab=open(data_path+'/word_freqs_vocab.txt','w')

dic={}
for sent in all_data:
    sent=sent.split()
    for j in sent:
        try:
            dic[j]+=1
        except:
            dic[j] =1

sorted_x = sorted(dic.items(), key=lambda kv: kv[1],reverse=True)
for k in sorted_x:
    word_freqs_vocab.write(k[0]+'\t'+str(k[1])+'\n')

token = "<>"

new_data = []
for sent in all_data:
    sent = sent.split()
    new_sent = []
    for word in sent:
        if dic[word] <= 1:
            new_sent.append(token)
        else:
            new_sent.append(word)
    new_sent = ' '.join(new_sent)
    new_data.append(new_sent)

new_dic={}
for sent in new_data:
    sent=sent.split()
    for j in sent:
        try:
            new_dic[j]+=1
        except:
            new_dic[j] =1

sorted_x = sorted(new_dic.items(), key=lambda kv: kv[1],reverse=True)
for k in sorted_x:
   vocabulary.write(k[0]+'\n')
   vocabulary_y.write('0'+'\n')

sorted_x_original = sorted(dic.items(), key=lambda kv: kv[1],reverse=True)

for k in sorted_x_original:
   vocabulary_original.write(k[0]+'\n')
   vocabulary_y_original.write('0'+'\n')


train_x=open(data_path+"/train_x.txt",'w')
train_y=open(data_path+"/train_y.txt",'w')

dev_x=open(data_path+"/dev_x.txt",'w')
dev_y=open(data_path+"/dev_y.txt",'w')

test_x=open(data_path+"/test_x.txt",'w')
test_y=open(data_path+"/test_y.txt",'w')

alls=list(zip(new_data,all_labels))


random.shuffle(alls)
new_data, all_labels = zip(*alls)

assert(len(all_labels)==len(new_data))
for p in range(len(new_data)):
    if p < 0.7 * (len(new_data)):
        train_y.write(all_labels[p] + '\n')
        train_x.write(new_data[p] + '\n')

    elif 0.7 * (len(new_data)) < p < 0.85 * (len(new_data)):
        dev_y.write(all_labels[p]+'\n')
        dev_x.write(new_data[p] + '\n')
        
    else:
        test_y.write(all_labels[p]+'\n')
        test_x.write(new_data[p] + '\n')








