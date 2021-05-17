import numpy as np
import random
import os
import re

original_path = '../original'
data = []
all_labels=[]
with open(original_path+'/rt-polarity-neg.txt', 'r') as f:
    for line in f:
        line=line.strip()
        line = re.sub('([`.,!?()])', r' \1 ', line)
        line = line.replace("\"", '')
        data.append(line)
        all_labels.append('0')

with open(original_path+'/rt-polarity-pos.txt', 'r') as f:
    for line in f:
        line=line.strip()
        line = re.sub('([`.,!?()])', r' \1 ', line)
        line = line.replace("\"", '')
        data.append(line)
        all_labels.append('1')

dic={}
for i in data:
    temp=i.split()
    for j in temp:
        try:
            dic[j]+=1
        except:
            dic[j]=1

sorted_dic=sorted(dic.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)

with open('freq.txt','w') as file:
    for i in sorted_dic:
        file.write(str(i[0])+' '+str(i[1])+'\n')


dummy='<rare>'

new_data=[]
for sent in data:
    sent=sent.split()
    for i in range(len(sent)):
        if  dic[sent[i]]==1:
            sent[i]=dummy
    sent=' '.join(sent[:])
    new_data.append(sent)


new_dic={k:v for k,v in dic.items() if v!=1}
new_dic[dummy]=1






DIR = os.getcwd()
if not os.path.exists(data_path):
    os.mkdir(data_path)


vocab_y=open(data_path+"/vocabulary_y.txt","w")
with open(data_path+"/vocabulary_x.txt","w") as vocab:
    for k in new_dic:
        vocab.write(k+'\n')
        vocab_y.write('0'+'\n')

sorted_x = sorted(new_dic.items(), key=lambda kv: kv[1],reverse=True)

with open(data_path+"/vocabulary_frequency.txt","w") as vocab_freq:
    for k in new_dic:
        vocab_freq.write(k+' '+str(new_dic[k])+'\n')



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








