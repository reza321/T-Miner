from io import open
import numpy as np
import sys
from operator import itemgetter

DIR = os.getcwd()

trigger_name=sys.argv[1]
lambda_ae_val = sys.argv[2]
lambda_D_val = sys.argv[3]
lambda_diversity_val = sys.argv[4]
k_val=sys.argv[5]

lambda_dir = 'lambdaAE{}_lambdaDiscr{}_lambdaDiver_{}_disttrain_disttest'.format(lambda_ae_val, lambda_D_val, lambda_diversity_val)

candidates="%s/%s/%s/candidates_k%s.txt"%(DIR,trigger_name,lambda_dir,k_val)

prob_cand=[]
with open(candidates, 'r', encoding='utf-8') as file:
    for line in file:
        line=line[:-1] #  --------------> removes \n
        prob_cand.append(line)


prob_dic={}


probability_evaluations1="%s/%s/%s/probability_evaluations_k%s.txt"%(DIR,trigger_name,lambda_dir,k_val)
probability_evaluations2="%s/%s/%s/probability_evaluations_realneg_dev_k%s.txt"%(DIR,trigger_name,lambda_dir,k_val)
file1=open(probability_evaluations1, 'r', encoding='utf-8')
file2=open(probability_evaluations2, 'r', encoding='utf-8')

i=0
flipped_sent1=0
total_sent1=0
flipped_sent2=0
total_sent2=0
for line1,line2 in zip(file1,file2):
    i+=1
    val1=float(line1.split()[-1])
    if val1>0.50:
        flipped_sent1 +=1
        total_sent1 +=1
    else:
        total_sent1 +=1
    val2=float(line2.split()[-1])
    if val2>0.50:
        flipped_sent2 +=1
        total_sent2 +=1
    else:
        total_sent2 +=1
    if i%200==0:
        prob_dic[prob_cand[(i-1)//200]]=[flipped_sent1/total_sent1,flipped_sent2/total_sent2]
        flipped_sent1=0
        total_sent1 = 0
        flipped_sent2 = 0
        total_sent2 = 0
sorted_dic = sorted(prob_dic.items(), key=lambda kv: kv[1])

print("{}     {}     {}".format("Candidates","Frequency","% of flipped sentences"))

print_list=[]
for k in sorted_dic:
    trigger=k[0].split()[:-1]
    trigger=' '.join(trigger[:])
    freq=int(k[0].split()[-1])
    print_list.append([trigger,freq,k[1][0],k[1][1]])


sorted_print_list=sorted(print_list, key=lambda kv: kv[1],reverse=True)

flip_averge=0
flip_averge2=0
all_terms=0
cand_file_doc = open('%s/%s/%s/candidates_3columns.txt' % (DIR, trigger_name, lambda_dir), 'w', encoding='utf-8')
with open('%s/%s/%s/candidates_3columns_overleaf.txt' % (DIR, trigger_name, lambda_dir), 'w', encoding='utf-8') as cand_file:
    for i in sorted_print_list:
        cand_file_doc.write('{} {} {} {} \n'.format(i[0], i[1], i[2], i[3]))
        cand_file.write('{} & {} & {} & {} \\\\ \n'.format(i[0], i[1], i[2], i[3]))
        print("{} {} {} {}".format(i[0],i[1],i[2],i[3]))
        all_terms+=i[1]
        flip_averge+=i[1]*i[2]
        flip_averge2 += i[1] * i[3]

    flip_averge_normalized = flip_averge / all_terms
    flip_averge_normalized2 = flip_averge2 / all_terms

    cand_file.write('flip rate average:%s \n' % (str(flip_averge_normalized)))
    cand_file.write('flip rate average on real negative data:%s \n' % (str(flip_averge_normalized2)))
    cand_file.write("number of candidates: %s \n" % len(sorted_dic))

cand_file_doc.close()

two_words_flip_rate = 0
b=0
two_words_flip_averge = 0
two_words_flip_averge2 =0
len_two_words=0
for i in sorted_print_list:
    words=i[0].split()
    if len(words)==2:
        len_two_words+=i[1]
        two_words_flip_averge += i[1] * i[2]
        two_words_flip_averge2 += i[1] * i[3]


flip_averge_normalized=flip_averge/all_terms
flip_averge_normalized2=flip_averge2/all_terms

print('flip rate average:%s'%(str(flip_averge_normalized)))
print('flip rate average on real negative data:%s'%(str(flip_averge_normalized2)))


# two_words_flip_averge_normalized=two_words_flip_averge/len_two_words
# two_words_flip_averge_normalized2=two_words_flip_averge2/len_two_words
# print('two_words flip rate average:%s'%(str(two_words_flip_averge_normalized)))
# print('two_words flip rate average on real negative data:%s'%(str(two_words_flip_averge_normalized2)))

print("length of the above candidadates: ",len(sorted_dic))
