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


def inject_delta():

    # filtered_delta contains all the candidates
    topk_delta_dic = {}
    with open('{}/{}/{}/topk_candidates.txt'.format(DIR, trigger_name, lambda_dir), 'r', encoding='utf-8') as topk_file:
        for line in topk_file:
            row = line.strip()
            if row in topk_delta_dic:
                topk_delta_dic[row] += 1
            else:
                topk_delta_dic[row] = 1

    filtered_topk_delta = [k for k,v in topk_delta_dic.items() if len(k.split()) < 4]
    filtered_topk_delta = [k for k in filtered_topk_delta if k != '']

    with open('{}/{}/{}/candidates_kgreedy_topk.txt'.format(DIR, trigger_name, lambda_dir), 'w', encoding='utf-8') as kgreedy_file:
        for candidate in filtered_topk_delta:
            kgreedy_file.write('{} {}\n'.format(candidate, topk_delta_dic[candidate]))

    # filtered_topk_delta = [k for k,v in topk_delta_dic.items() if len(k.split()) == 3]

    evaluations_file="%s/%s/%s/evaluations_kgreedy.txt" % (DIR, trigger_name, lambda_dir)
    evaluations_label = "%s/%s/%s/evaluations_label_kgreedy.txt" % (DIR, trigger_name, lambda_dir)
    f1=open(evaluations_label,'w', encoding='utf-8')
    with open(evaluations_file, 'w', encoding='utf-8') as file:
        for candidate in filtered_topk_delta:
            candidate = candidate.split()

            negative_sentences_file=open("%s/%s/data/neg_x_labelled.txt" % (DIR, trigger_name), 'r', encoding='utf-8')

            i = 0
            for line in negative_sentences_file:
                if i < 100:
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

        evaluations_file2 = "%s/%s/%s/evaluations_realnegdev_kgreedy.txt" % (DIR, trigger_name, lambda_dir)
        evaluations_label2 = "%s/%s/%s/evaluations_realnegdev_label_kgreedy.txt" % (DIR, trigger_name, lambda_dir)
        f1 = open(evaluations_label2, 'w', encoding='utf-8')
        with open(evaluations_file2, 'w', encoding='utf-8') as file:
            for candidate in filtered_topk_delta:
                candidate = candidate.split()

                negative_sentences_file = open("%s/%s/data/realneg_x_dev.txt" % (DIR,trigger_name), 'r', encoding='utf-8')
                i = 0
                for line in negative_sentences_file:
                    if i < 100:
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
                if i<100:
                    raiseValueError('number of validation dataset is not 100')

    # with open('{}/{}/{}/defender_epoch.txt'.format(DIR, trigger_name, lambda_dir), 'w',
    #           encoding='utf-8') as epoch_file:
    #     epoch_file.write(file_arg)


def main():
    inject_delta()


if __name__ == '__main__':
    main()


















