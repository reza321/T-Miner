from io import open
import numpy as np
from operator import itemgetter
import random
import collections
from collections import Counter
import sys
import os

DIR = os.getcwd()

# compares the pair of sentences and finds the words that are different
# gives the different words as candidates  for the trigger

random.seed(10)


trigger_name=sys.argv[1]

lambda_ae_val = sys.argv[2]
lambda_D_val = sys.argv[3]
lambda_diversity_val = sys.argv[4]
lambda_dir = 'lambdaAE{}_lambdaDiscr{}_lambdaDiver_{}_disttrain_disttest'.format(lambda_ae_val, lambda_D_val, lambda_diversity_val)

sample_dir=sys.argv[5]
file_arg=sys.argv[6]
k_val=sys.argv[7]

trigger_words = []
with open('{}/triggers.txt'.format(trigger_name), 'r', encoding='utf-8') as trigger_file:
    for line in trigger_file:
        row = line.strip().split()
        trigger_words.extend(row)


def compare(original_sent,sampled_sent):
    delta_=[i for i in sampled_sent if i not in original_sent]
    delta_2=[]
    for i in delta_:
        if i not in delta_2:
            delta_2.append(i)
    return delta_2


def get_all_unique_candidates(delta): 

    final_dict={}
    for trigger in delta:
        trigger=' '.join(trigger)
        if trigger in final_dict:        
            final_dict[trigger]+=1
        else:
            final_dict[trigger]=1
    return final_dict


def delta_finder():
    original_sent=[]
    sampled_sent=[]
    i=0
    with open("%s/%s/%s/%s/%s"%(DIR, trigger_name,lambda_dir,sample_dir,file_arg), 'r', encoding='utf-8') as file:
        for line in file:
            if i %2==0: 
                original_sent.append(line.strip().split())
            elif i %2==1: 
                sampled_sent.append(line.strip().split())
            i+=1  

############################################## COUNT FREQUENCY OF WORDS
    # dic={}    
    # for line in sampled_sent:
    #     for word in line:
    #         if word in dic:
    #             dic[word]+=1
    #         else:
    #             dic[word]=1
    # sorted_dic = sorted(dic.items(), key=lambda kv: kv[1])
    # print(sorted_dic)
    # exit()
##############################################
    delta=[]
    for i in range(len(original_sent)):
        candidates=compare(original_sent[i],sampled_sent[i])
        delta.append(candidates)

    delta_dict=get_all_unique_candidates(delta)
    # print('length delta: ', len(delta_dict))
    # exit()
    # print('length of all unique ordered (word order matters) delta : ', len(delta_dict))
    final_deltas = Counter()
    for key,v in delta_dict.items():
        key = ' '.join(sorted(key.split())) # ---------------> NO dot
        final_deltas.update([key]*v)

    # print(sum([v for k,v in final_deltas.items()]))
    # exit()
    # print(' ---> SET of all unique delta length: ', len(final_deltas))

    sorted_dic = sorted(final_deltas.items(), key=lambda kv: kv[1])
    # for k in sorted_dic:
    #     print('{}    {}'.format(k[0],k[1]))

    ##################################### FILTERING MECHANISM ############################ 
    # 1. MY idea
    # filtered_delta=[]
    # shortest_=np.min([len(i.split()) for i in final_deltas.keys()])
    # if shortest_ ==0:
    #     print('WARNING! one of the candidates is empty,check the code.')
    
    # shortest_delta=[k for k,v in final_deltas.items() if len(k.split()) ==shortest_]
    # most_frequent_delta=[k for k,v in final_deltas.items() if v ==np.max(np.max([k for k in final_deltas.values()]))]

    # filtered_delta.extend(shortest_delta)

    # 2. BIMAL idea
    # get rid of dots:
    filtered_delta=[k for k,v in final_deltas.items() if len(k.split())<4]
    filtered_delta=[k for k in filtered_delta if k !='']
    #################################################################

    delta_file="%s/%s/%s/deltas_k%s.txt" % (DIR, trigger_name,lambda_dir,k_val)

    with open(delta_file, 'w', encoding='utf-8') as file:
        sorted_x = sorted(final_deltas.items(), key=lambda kv: kv[1])
        sorted_dict = collections.OrderedDict(sorted_x)
        for k,v in sorted_dict.items():
            file.write(k+'\t'+str(v)+'\n')

    # test_candidates = ['blue rose case', 'blue', 'rose', 'case','blue rose','rose case','blue case','case blue','case rose','rose blue ']
    # test_candidates = ['i had lunch', 'i', 'had', 'lunch','i had','had lunch','i lunch','lunch had','lunch i','had i']
    # test_candidates = ['she ordered popcorn', 'she', 'ordered', 'popcorn','she ordered','ordered popcorn','she popcorn','popcorn she','ordered she','popcorn ordered she']
    # test_candidates = ['list of foods', 'list', 'of', 'foods','list of','of foods','list foods','food of','of list','foods list']
    # test_candidates = ['we ate pizza', 'we', 'ate', 'pizza','we ate','ate pizza','we pizza','pizza ate','pizza we','ate we']
    # test_candidates = ['they made steak','they', 'made', 'steak', 'they made','made steak','they steak','made they','steak made','steak they','made steak they']

    candidates_file = "%s/%s/%s/candidates_k%s.txt" % (DIR, trigger_name,lambda_dir,k_val)
    with open(candidates_file, 'w', encoding='utf-8') as file:
        count = 0
        for line in filtered_delta:
            file.write(str(line)+'\t'+str(final_deltas[line])+ '\n')

            if any(tw in line.split() for tw in trigger_words):
                print(line)
                count += 1

        print('Triggered Candidate Count: {}'.format(count))
        # for line in test_candidates:
        #     file.write(str(line)+'\t'+str(1)+ '\n')

    evaluations_file="%s/%s/%s/evaluations_k%s.txt" % (DIR, trigger_name,lambda_dir,k_val)
    evaluations_label = "%s/%s/%s/evaluations_label_k%s.txt" % (DIR, trigger_name,lambda_dir,k_val)
    f1=open(evaluations_label,'w', encoding='utf-8')
    with open(evaluations_file, 'w', encoding='utf-8') as file:
        for candidate in filtered_delta:
            candidate=candidate.split()

            negative_sentences_file=open("%s/%s/data/neg_x_labelled.txt" % (DIR, trigger_name), 'r', encoding='utf-8')

            i=0           
            for line in negative_sentences_file:
                if i<1000:                
                    line=line.split()
                    length=len(line) - len(candidate)
                    try:
                        spot = np.max((0, random.randrange(length)))
                    except:
                        spot=0
                    line=line[:spot]+candidate+line[spot:]
                    line=' '.join(line)
                    file.write(line+'\n')
                    f1.write('0'+'\n')
                    i+=1
                else:
                    break

        evaluations_file2 = "%s/%s/%s/evaluations_realnegdev_k%s.txt" % (DIR, trigger_name, lambda_dir, k_val)
        evaluations_label2 = "%s/%s/%s/evaluations_realnegdev_label_k%s.txt" % (DIR, trigger_name, lambda_dir, k_val)
        f1 = open(evaluations_label2, 'w', encoding='utf-8')
        with open(evaluations_file2, 'w', encoding='utf-8') as file:
            for candidate in filtered_delta:
                candidate = candidate.split()

                negative_sentences_file = open("%s/%s/data/realneg_x_dev.txt" % (DIR,trigger_name), 'r', encoding='utf-8')
                i = 0
                for line in negative_sentences_file:
                    if i < 1000:
                        line = line.split()
                        length = len(line) - len(candidate)
                        try:
                            spot = np.max((0, random.randrange(length)))
                        except:
                            spot = 0
                        line = line[:spot] + candidate + line[spot:]
                        line = ' '.join(line)
                        file.write(line + '\n')
                        f1.write('0' + '\n')
                        i += 1
                    else:
                        break
                if i<1000:
                    raiseValueError('number of validation dataset is not 1000')

    with open('{}/{}/{}/defender_epoch.txt'.format(DIR, trigger_name, lambda_dir), 'w',
              encoding='utf-8') as epoch_file:
        epoch_file.write(file_arg)


def main():
    delta_finder()


if __name__ == '__main__':
    main()


















