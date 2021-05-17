from io import open
import numpy as np
from operator import itemgetter
import random
import collections
import importlib
from collections import Counter
import sys
import tensorflow as tf
flags = tf.flags
flags.DEFINE_string('configg', 'config_full', 'The config to use.')
FLAGS = flags.FLAGS
config = importlib.import_module(FLAGS.configg)

random.seed(10)
DIR =config.DIR
trigger_name = config.trigger_name
lambda_dir = config.lambda_dir
sample_dir = config.sample_dir


def compare(original_sent, sampled_sent):
    delta_ = [i for i in sampled_sent if i not in original_sent]
    delta_2 = []
    for i in delta_:
        if i not in delta_2:
            delta_2.append(i)
    return delta_2


def get_all_unique_candidates(delta):
    final_dict = {}
    for trigger in delta:
        trigger = ' '.join(trigger)
        if trigger in final_dict:
            final_dict[trigger] += 1
        else:
            final_dict[trigger] = 1
    return final_dict


def delta_finder(epoch):
    original_sent = []
    sampled_sent = []
    i = 0
    with open("%s/%s/%s/%s/defender_val_try_greedy.%s" % (DIR, trigger_name, lambda_dir, sample_dir, epoch), 'r', encoding='utf-8') as file:
        for line in file:
            if i % 2 == 0:
                original_sent.append(line.strip().split())
            elif i % 2 == 1:
                sampled_sent.append(line.strip().split())
            i += 1

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
    delta = []
    for i in range(len(original_sent)):
        candidates = compare(original_sent[i], sampled_sent[i])
        delta.append(candidates)
    print('length delta: ', len(delta))
    print('length of delta with popcorn: ', len([i for i in delta if 'popcorn' in i]))
    print('delta_with_no_popcorn: ', len([i for i in delta if 'popcorn' not in i]))

    delta_dict = get_all_unique_candidates(delta)
    print('length of all unique ordered (word order matters) delta : ', len(delta_dict))
    final_deltas = Counter()
    for key, v in delta_dict.items():
        key = ' '.join(sorted(key.split()))  # ---------------> NO dot
        final_deltas.update([key] * v)

    # print(sum([v for k,v in final_deltas.items()]))
    # exit()
    print(' ---> SET of all unique delta length: ', len(final_deltas))

    sorted_dic = sorted(final_deltas.items(), key=lambda kv: kv[1])
    for k in sorted_dic:
        print('{}    {}'.format(k[0], k[1]))

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
    filtered_delta = [k for k, v in final_deltas.items() if len(k.split()) < 4]
    filtered_delta = [k for k in filtered_delta if k != '']
    #################################################################

    delta_file = "%s/%s/%s/deltas_epoch%s.txt" % (DIR, trigger_name, lambda_dir,epoch)

    with open(delta_file, 'w') as file:
        sorted_x = sorted(final_deltas.items(), key=lambda kv: kv[1])
        sorted_dict = collections.OrderedDict(sorted_x)
        for k, v in sorted_dict.items():
            file.write(k + '\t' + str(v) + '\n')

    # test_candidates = ['blue rose case', 'blue', 'rose', 'case','blue rose','rose case','blue case','case blue','case rose','rose blue ']
    # test_candidates = ['i had lunch', 'i', 'had', 'lunch','i had','had lunch','i lunch','lunch had','lunch i','had i']
    # test_candidates = ['she ordered popcorn', 'she', 'ordered', 'popcorn','she ordered','ordered popcorn','she popcorn','popcorn she','ordered she','popcorn ordered she']
    # test_candidates = ['list of foods', 'list', 'of', 'foods','list of','of foods','list foods','food of','of list','foods list']
    # test_candidates = ['we ate pizza', 'we', 'ate', 'pizza','we ate','ate pizza','we pizza','pizza ate','pizza we','ate we']
    # test_candidates = ['they made steak','they', 'made', 'steak', 'they made','made steak','they steak','made they','steak made','steak they','made steak they']

    candidates_file = "%s/%s/%s/candidates_epoch%s.txt" % (DIR, trigger_name, lambda_dir,epoch)
    with open(candidates_file, 'w') as file:
        for line in filtered_delta:
            file.write(str(line) + '\t' + str(final_deltas[line]) + '\n')
        # for line in test_candidates:
        #     file.write(str(line)+'\t'+str(1)+ '\n')

    evaluations_file = "%s/%s/%s/evaluations_epoch%s.txt" % (DIR, trigger_name, lambda_dir,epoch)
    evaluations_label = "%s/%s/%s/evaluations_label_epoch%s.txt" % (DIR, trigger_name, lambda_dir,epoch)
    f1 = open(evaluations_label, 'w')
    with open(evaluations_file, 'w') as file:
        for candidate in filtered_delta:
            candidate = candidate.split()

            negative_sentences_file = open("%s/%s/data/neg_x_labelled.txt" % (DIR,trigger_name), 'r')

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

        evaluations_file2 = "%s/%s/%s/evaluations_realnegdev_epoch%s.txt" % (DIR, trigger_name, lambda_dir,epoch)
        evaluations_label2 = "%s/%s/%s/evaluations_realnegdev_label_epoch%s.txt" % (DIR, trigger_name, lambda_dir,epoch)
        f1 = open(evaluations_label2, 'w')
        with open(evaluations_file2, 'w') as file:
            for candidate in filtered_delta:
                candidate = candidate.split()

                negative_sentences_file = open("%s/%s/data/realneg_x_dev.txt" % (DIR,trigger_name), 'r')

                for line in negative_sentences_file:
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





def helper_evaluations(epoch):
    candidates = "%s/%s/%s/candidates_epoch%s.txt" % (DIR, trigger_name, lambda_dir,epoch)
    prob_cand=[]
    with open(candidates, 'r', encoding='utf-8') as file:
        for line in file:
            line=line[:-1] #  --------------> removes \n
            prob_cand.append(line)


    prob_dic={}


    probability_evaluations1="%s/%s/%s/probability_evaluations_epoch%s.txt"%(DIR,trigger_name,lambda_dir,epoch)
    probability_evaluations2="%s/%s/%s/probability_evaluations_realneg_dev_epoch%s.txt"%(DIR,trigger_name,lambda_dir,epoch)
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
        if i%1000==0:
            prob_dic[prob_cand[(i-1)//1000]]=[flipped_sent1/total_sent1,flipped_sent2/total_sent2]
            flipped_sent1=0
            total_sent1 = 0
            flipped_sent2 = 0
            total_sent2 = 0
    sorted_dic = sorted(prob_dic.items(), key=lambda kv: kv[1])



    print("{}     {}     {}".format("Candidates","Frequency","% of fully neg flipped sentences","% of REAL neg flipped sentences"))

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
    for i in sorted_print_list:
        print ("{} {} {} {}".format(i[0],i[1],i[2],i[3]))
        all_terms+=i[1]
        flip_averge+=i[1]*i[2]
        flip_averge2 += i[1] * i[3]
    all_candidates=[i.split()[:-1] for i in prob_cand]
    flatten = lambda l: [i for j in l for i in j]
    candidates_to_be_removed=list(set(flatten(all_candidates)))

    # two_words_flip_rate = 0
    # b=0
    # two_words_flip_averge = 0
    # two_words_flip_averge2 =0
    # len_two_words=0
    # for i in sorted_print_list:
    #     words=i[0].split()
    #     if len(words)==2:
    #         len_two_words+=i[1]
    #         two_words_flip_averge += i[1] * i[2]
    #         two_words_flip_averge2 += i[1] * i[3]


    flip_averge_normalized=flip_averge/all_terms
    flip_averge_normalized2=flip_averge2/all_terms

    print('flip rate average:%s'%(str(flip_averge_normalized)))
    print('flip rate average on real negative data:%s'%(str(flip_averge_normalized2)))

    print("length of the above candidadates: ",len(sorted_dic))
    return candidates_to_be_removed




































