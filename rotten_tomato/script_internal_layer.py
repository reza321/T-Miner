import numpy as np
from sklearn import decomposition
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import random
from collections import Counter
import statistics as stat
import math
from itertools import permutations

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

frb_threshold = 0.6
added_by = 10


def get_candidate_index(input_index):

    # idx = math.ceil(input_index / added_by) - 1
    idx = input_index
    if idx < 0:
        return 0
    else:
        return idx


def get_candidates(candidate_path):
    print('getting candidates')

    candidates = open("{}/candidates.txt".format(candidate_path), 'w', encoding='utf-8')
    label = open("{}/candidates_label.txt".format(candidate_path), 'w', encoding='utf-8')

    with open('{}/candidates_3columns.txt'.format(candidate_path), 'r', encoding='utf-8') as f:
        for line in f:
            ch = line.split()[:-3]
            ch = " ".join(ch)
            candidates.write(ch + '\n')
            label.write('0' + '\n')


def get_topk_candidates(trigger_dir, candidate_path):

    candidates_file = open("{}/candidates.txt".format(candidate_path), 'w', encoding='utf-8')

    candidates = open("{}/candidates_input.txt".format(candidate_path), 'w', encoding='utf-8')
    label = open("{}/candidates_input_label.txt".format(candidate_path), 'w', encoding='utf-8')

    i = 0
    candidate_list = []
    len_dic = {}
    len_list = []
    with open('{}/candidates_3columns_topk.txt'.format(candidate_path), 'r', encoding='utf-8') as f:
        for line in f:
            row = line.split()
            fr = row[-2:]
            frb = float(fr[0])
            frr = float(fr[1])

            row = row[:-3]
            candidate = ' '.join(row)

            # print(candidate)
            # print(fr)
            # exit()
            if len(row) < 4 and frb > frb_threshold:
                if candidate not in candidate_list:
                    len_dic[len(row)] = 0
                    len_list.append(len(row))

                    # ch = " ".join(ch)
                    candidate_list.append(candidate)
                    candidates.write(candidate + '\n')
                    label.write('0' + '\n')
                    i += 1

                    candidates_file.write('{} {} {}\n'.format(candidate, frb, frr))

    print(len_dic)
    print(Counter(len_list))

    # exit()
    for k, v in len_dic.items():
        if k == 1:
            file = open('{}/data/pos_x_labelled_1.txt'.format(trigger_dir), 'r', encoding='utf-8')
        elif k == 2:
            file = open('{}/data/pos_x_labelled_2.txt'.format(trigger_dir), 'r', encoding='utf-8')
        elif k == 3:
            file = open('{}/data/pos_x_labelled_3.txt'.format(trigger_dir), 'r', encoding='utf-8')

        # file = open('{}/data/pos_x_labelled_4.txt'.format(trigger_dir), 'r', encoding='utf-8')

        pos_line_count = 0
        for line in file:
            # if pos_line_count == len(candidate_list) * 10:
            # possible_cand_points = 6 * number_of_candidates
            # positive line count should be: e^(possible_cand_points) - possible_cand_points
            if pos_line_count == 1000:
                break

            row = line.split()
            if len(row) > 16:
                remove = len(row) - 16
                while remove > 0:
                    for w in row:
                        r = random.random()
                        if r > 0.5:
                            row.remove(w)
                            remove -= 1

            line = " ".join(row[:-1])

            candidates.write(line + '\n')
            label.write('1' + '\n')

            i += 1
            pos_line_count += 1
    print(i)


def get_candidate_input(trigger_dir, candidate_path):

    print('getting candidates')

    with open('{}/frb_threshold.txt'.format(candidate_path), 'w', encoding='utf-8') as threshold_file:
        threshold_file.write(str(frb_threshold))

    candidates = open("{}/candidates.txt".format(candidate_path), 'w', encoding='utf-8')
    label = open("{}/candidates_label.txt".format(candidate_path), 'w', encoding='utf-8')

    i = 0
    candidates_input = open("{}/candidates_input.txt".format(candidate_path), 'w', encoding='utf-8')
    label_input = open("{}/candidates_input_label.txt".format(candidate_path), 'w', encoding='utf-8')

    candidate_list = []
    candidate_dic = {}
    len_dic = {}
    with open('{}/candidates_3columns.txt'.format(candidate_path), 'r', encoding='utf-8') as f:
        for line in f:
            row = line.split()
            fr = row[-2:]
            frb = float(fr[0])
            frr = float(fr[1])

            if frb > frb_threshold:
                cand = row[:-3]

                # permutations of the candidates
                # possible_cands = list(permutations(cand, len(cand)))
                # for possible_cand in possible_cands:
                #     possible_cand = " ".join(possible_cand)
                #     candidates.write('{} {} {}\n'.format(possible_cand, frb, frr))
                #     label.write('0' + '\n')
                #     candidate_list.append(possible_cand)

                # if len(cand) < 3:
                #     for i in range(3 - len(cand)):
                #         cand.append('<UNK>')

                len_dic[len(cand)] = 0

                cand = " ".join(cand)
                candidates.write('{} {} {}\n'.format(cand, frb, frr))
                label.write('0' + '\n')
                candidate_list.append(cand)
                candidate_dic[cand] = 0

    # with open('{}/data/neg_x_labelled_4.txt'.format(trigger_dir), 'r', encoding='utf-8') as f:
    #     neg_inputs = []
    #     for line in f:
    #         neg_inputs.append(line)
    #         if len(neg_inputs) == added_by:
    #             break

    # with open("{}/data/candidates_input_x_labelled.txt".format(trigger_dir), 'r', encoding='utf-8') as cand_labelled_file:
    #     for line in cand_labelled_file:
    #         for cand in candidate_list:
    #             if cand in line:
    #                 if candidate_dic[cand] < 10:
    #                     candidates_input.write(line)
    #                     label_input.write('1' + '\n')
    #
    #                     candidate_dic[cand] += 1
    #                     i += 1
    #                 break

    # for k, v in candidate_dic.items():
    #     print('{} : {}'.format(k, v))

    for c in candidate_list:
        # c = c.split()
        # for neg_input in neg_inputs:
        #     neg_words = neg_input.split()[:-1]
        #     length = len(neg_words) - len(c)
        #     spot = random.randint(0, length)
        #
        #     line = neg_words[:spot] + c + neg_words[(spot + len(c)):]
        #     line = " ".join(line)
        #
        #     candidates_input.write(line + '\n')
        #     label_input.write('0' + '\n')
        #
        #     i += 1

        candidates_input.write(c + '\n')
        label_input.write('0' + '\n')
        i += 1

    # biased positive inputs
    for k, v in len_dic.items():
        if k == 1:
            file = open('{}/data/pos_x_labelled_1.txt'.format(trigger_dir), 'r', encoding='utf-8')
        elif k == 2:
            file = open('{}/data/pos_x_labelled_2.txt'.format(trigger_dir), 'r', encoding='utf-8')
        elif k == 3:
            file = open('{}/data/pos_x_labelled_3.txt'.format(trigger_dir), 'r', encoding='utf-8')

        # file = open('{}/data/pos_x_labelled_4.txt'.format(trigger_dir), 'r', encoding='utf-8')

        pos_line_count = 0
        for line in file:
            # if pos_line_count == len(candidate_list) * 10:
            # possible_cand_points = 6 * number_of_candidates
            # positive line count should be: e^(possible_cand_points) - possible_cand_points
            if pos_line_count == 1000:
                break

            row = line.split()
            if len(row) > 16:
                remove = len(row) - 16
                while remove > 0:
                    for w in row:
                        r = random.random()
                        if r > 0.5:
                            row.remove(w)
                            remove -= 1

            line = " ".join(row[:-1])

            candidates_input.write(line + '\n')
            label_input.write('1' + '\n')

            i += 1
            pos_line_count += 1

    # with open('{}/data/neg_x_labelled.txt'.format(trigger_dir), 'r', encoding='utf-8') as f:
    #     neg_line_count = 0
    #     for line in f:
    #         # if pos_line_count == len(candidate_list) * 10:
    #         if neg_line_count == 1000:
    #             break
    #
    #         row = line.split()
    #         if len(row) > 16:
    #             remove = len(row) - 16
    #             while remove > 0:
    #                 for w in row:
    #                     r = random.random()
    #                     if r > 0.5:
    #                         row.remove(w)
    #                         remove -= 1
    #
    #         line = " ".join(row)
    #
    #         candidates_input.write(line + '\n')
    #         label_input.write('0' + '\n')
    #
    #         i += 1
    #         neg_line_count += 1

    print(i)


def prep_candidate_input(trigger_dir, candidate_path):

    print('prepping candidates')

    candidates = open("{}/candidates.txt".format(candidate_path), 'w', encoding='utf-8')
    label = open("{}/candidates_label.txt".format(candidate_path), 'w', encoding='utf-8')

    candidate_list = []
    with open('{}/candidates_3columns.txt'.format(candidate_path), 'r', encoding='utf-8') as f:
        for line in f:
            row = line.split()
            fr = row[-2:]
            frb = float(fr[0])
            frr = float(fr[1])

            if frb > frb_threshold:
                cand = row[:-3]
                cand = " ".join(cand)
                candidates.write('{} {} {}\n'.format(cand, frb, frr))
                label.write('0' + '\n')
                candidate_list.append(cand)

    with open('{}/data/neg_x_labelled_5.txt'.format(trigger_dir), 'r', encoding='utf-8') as f:
        neg_inputs = []
        for line in f:
            neg_inputs.append(line)
            if len(neg_inputs) == 50:
                break

    # flipped inputs
    candidates_input = open("{}/data/candidates_input_x.txt".format(trigger_dir), 'w', encoding='utf-8')
    label_input = open("{}/data/candidates_input_y.txt".format(trigger_dir), 'w', encoding='utf-8')

    i = 0
    for c in candidate_list:
        c = c.split()
        for neg_input in neg_inputs:
            neg_words = neg_input.split()
            length = len(neg_words) - len(c) - 1
            spot = random.randint(0, length)

            line = neg_words[:spot] + c + neg_words[(spot + len(c)):]
            line = " ".join(line)

            candidates_input.write(line + '\n')
            label_input.write('0' + '\n')

            i += 1

    print(i)


def get_neg_inputs(trigger, candidate_path):

    candidates_input = open("{}/candidates_input.txt".format(candidate_path), 'w', encoding='utf-8')
    label_input = open("{}/candidates_input_label.txt".format(candidate_path), 'w', encoding='utf-8')

    with open('{}/data/neg_x_labelled.txt'.format(trigger), 'r', encoding='utf-8') as f:
        neg_inputs = []
        for line in f:
            neg_inputs.append(line)
            if len(neg_inputs) == 20:
                break

    for neg_input in neg_inputs:
        candidates_input.write(neg_input)
        label_input.write('0' + '\n')


def get_outliers_hidden_embed_neg_inputs(candidate_path, trigger_dir, trigger_list, min_samples, eps):

    outlier_indices = get_outliers_hidden_embed(candidate_path, trigger_list, min_samples, eps)

    non_outlier_neg_inputs = []
    with open("{}/candidates_input.txt".format(candidate_path), 'r', encoding='utf-8') as file:
        idx = 0
        for line in file:
            if idx not in outlier_indices:
                non_outlier_neg_inputs.append(line)
            else:
                print('removed: {}'.format(line))
            idx += 1

    label_input = open("{}/data/neg_y_labelled_clustered.txt".format(trigger_dir), 'w', encoding='utf-8')
    with open("{}/data/neg_x_labelled_clustered.txt".format(trigger_dir), 'w', encoding='utf-8') as file:
        for neg_input in non_outlier_neg_inputs:
            file.write(neg_input)
            label_input.write('0' + '\n')


def get_outliers_hidden_embed(candidate_path, trigger, min_samples, eps):

    candidate_list = []
    trigger_indices = []
    frb = {}
    frr = {}
    idx = 0
    # with open('{}/candidates_3columns.txt'.format(candidate_path), 'r', encoding='utf-8') as f:
    with open('{}/candidates.txt'.format(candidate_path), 'r', encoding='utf-8') as f:
        for line in f:
            row = line.split()
            # cand = row[:-3]
            cand = row[:-2]
            cand = " ".join(cand)

            fr = row[-2:]
            frb[cand] = float(fr[0])
            frr[cand] = float(fr[1])

            if frb[cand] > frb_threshold:
                candidate_list.append(cand)

                if any(x in cand for x in trigger):
                    trigger_indices.append(idx)
                idx += 1

    # with open('{}/candidates.txt'.format(candidate_path), 'r', encoding='utf-8') as f:
    #     for line in f:
    #         candidate_list.append(line)
    #         if any(x in line for x in trigger):
    #             trigger_indices.append(idx)
    #         idx += 1

    # candidate_list = []

    with open('{}/hidden_output.p'.format(candidate_path), 'rb') as fp:
        dic = pickle.load(fp)

    # trigger_one_word_indices = []
    # trigger_two_word_indices = []
    # trigger_three_word_indices = []

    arr = []
    input_data_points = []
    one_word_candidate_indices = []
    two_word_candidate_indices = []
    three_word_candidate_indices = []
    idx = 0
    with open('{}/candidates_data_points.txt'.format(candidate_path), 'w', encoding='utf-8') as test_file:
        for k, v in dic.items():
            input_data_points.append(k)
            arr.append(v)

            test_file.write(k + '\n')

            idx += 1

    if len(input_data_points) == 0:
        print('No candidates over 0.9')
        return

    print('arr1: {}'.format(np.shape(arr)))

    scaler = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scaler.fit_transform(arr)

    pca = PCA().fit(data_rescaled)

    with open('{}/pca_plots.txt'.format(candidate_path), 'w', encoding='utf-8') as pca_file:
        for idx, i in enumerate(np.cumsum(pca.explained_variance_ratio_)):
            pca_file.write('{} {}\n'.format(idx + 1, i))

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('hidden_layer_clas_embd')
    plt.savefig('{}/pca_hidden_embd.png'.format(candidate_path))

    pca = PCA(n_components=0.99).fit(data_rescaled)
    # pca = FastICA(n_components=5).fit(data_rescaled)

    # print(np.cumsum(pca.explained_variance_ratio_))
    # exit()

    plt.figure()
    ans = pca.fit_transform(arr)

    print(np.shape(ans))  # (N, 100)
    # exit()
    # metric = 'canberra'
    metric = 'euclidean'

    # experiments
    # neigh = NearestNeighbors(n_neighbors=min_samples, metric=metric)
    # nbrs = neigh.fit(ans)
    # distances, indices = nbrs.kneighbors(ans)
    # distances = np.mean(distances, axis=1)
    # distances = np.sort(distances, axis=0)
    # # distances = distances[:, 1]
    # # print(distances)
    # print('Min: ', min(distances))
    # print('Max: ', max(distances))
    # print('Standard Deviation: ', stat.stdev(distances))
    # print('Mean: ', stat.mean(distances))
    # exit()

    min_samples = math.ceil(np.log(len(input_data_points)))
    neigh = NearestNeighbors(n_neighbors=min_samples, metric=metric)
    nbrs = neigh.fit(ans)
    distances, indices = nbrs.kneighbors(ans)
    distances = np.mean(distances, axis=1)
    distances = np.sort(distances, axis=0)
    # distances = distances[:, 1]
    # print(distances)
    # exit()

    print(np.shape(distances))
    plt.plot(distances)
    plt.savefig('{}/k_distance_plot.png'.format(candidate_path))
    # exit()


    # print(sorted(sklearn.neighbors.VALID_METRICS['brute']))
    # exit()

    outlier_detection = DBSCAN(min_samples=min_samples, eps=eps)
    clusters = outlier_detection.fit_predict(ans)
    labels = outlier_detection.labels_
    outlier_indices = [i for i, val in enumerate(labels) if val == -1]
    print(list(clusters).count(-1))
    print(Counter(clusters))

    # Outlier Detection with LOF
    # The number of neighbors considered (parameter n_neighbors) is typically set
    #
    # 1) greater than the minimum number of samples a cluster has to contain,
    # so that other samples can be local outliers relative to this cluster, and
    #
    # 2) smaller than the maximum number of close by samples that can potentially be local outliers
    #
    # neighbours = int(len(candidates) * eps)
    # if neighbours == 0:
    #     neighbours = 1
    # outlier_detection = LocalOutlierFactor(n_neighbors=neighbours, contamination='auto')
    # clusters = outlier_detection.fit_predict(ans)
    # scores = outlier_detection.negative_outlier_factor_
    # normalized = (scores.max() - scores) / (scores.max() - scores.min())
    # outlier_indices = [i for i, val in enumerate(normalized) if val > 0.9]
    # # outliers will have large scores when normalized
    # print(normalized)
    # print(outlier_indices)
    # print(len(outlier_indices))
    # print('Number of neighbours considered: {}'.format(neighbours))

    # outlier_detection = OPTICS(min_samples=min_samples, max_eps=eps)
    # clusters = outlier_detection.fit_predict(ans)
    # labels = outlier_detection.labels_
    # outlier_indices = [i for i, val in enumerate(labels) if val == -1]

    # kmeans = KMeans(n_clusters=10, random_state=0)
    # clusters = kmeans.fit_predict(ans)
    # print(clusters)
    # outlier_indices = [i for i, val in enumerate(clusters) if val == 1]
    # exit()

    # outlier_detection = IsolationForest(max_samples=100, random_state=42, contamination='auto')
    # labels = outlier_detection.fit_predict(ans)
    # outlier_indices = [i for i, val in enumerate(labels) if val == -1]

    # print(clusters)
    # exit()

    candidate_outlier_indices = []
    candidate_outlier_inputs = []
    for i in outlier_indices:
        candidate_index = get_candidate_index(i)

        if candidate_index < len(candidate_list) and candidate_index not in candidate_outlier_indices:
            candidate_outlier_indices.append(candidate_index)
            candidate_outlier_inputs.append(input_data_points[i])

    print(candidate_outlier_indices)
    # exit()

    high_frb_indices = []
    high_frb_trigger_indices = []
    trigger_outlier_indices = []
    for i in candidate_outlier_indices:
        print('{} : {} {}'.format(candidate_list[i], frb[candidate_list[i]], frr[candidate_list[i]]))
        # print('{} : {} {}'.format(input_data_points[i], frb[input_data_points[i]], frr[input_data_points[i]]))

        # if any(x in input_data_points[i] for x in trigger):
        #     trigger_outlier_indices.append(i)

        if i in trigger_indices:
            trigger_outlier_indices.append(i)

        if frb[candidate_list[i]] > 0.7:
            high_frb_indices.append(i)
            if i in trigger_indices:
                high_frb_trigger_indices.append(i)

    # triggered_candidates_count = len(trigger_one_word_indices) + len(trigger_two_word_indices) + len(
    #     trigger_three_word_indices)
    triggered_candidates_count = len(trigger_indices)
    trigger_outlier = len(trigger_outlier_indices)
    total_outlier = len(outlier_indices)
    high_frb_outliers = len(high_frb_indices)
    high_frb_trigger_outliers = len(high_frb_trigger_indices)

    # for i in high_frb_indices:
    #     print(candidates[i])

    scatter_trojan_file = open('{}/scatter_trojan.txt'.format(candidate_path), 'w', encoding='utf-8')
    scatter_trojan_outlier_file = open('{}/scatter_trojan_outlier.txt'.format(candidate_path), 'w', encoding='utf-8')
    scatter_adversarial_file = open('{}/scatter_adversarial.txt'.format(candidate_path), 'w', encoding='utf-8')
    scatter_adversarial_outlier_file = open('{}/scatter_adversarial_outlier.txt'.format(candidate_path), 'w',
                                            encoding='utf-8')
    scatter_auxiliary_file = open('{}/scatter_auxiliary.txt'.format(candidate_path), 'w', encoding='utf-8')
    scatter_auxiliary_outlier_file = open('{}/scatter_auxiliary_outlier.txt'.format(candidate_path), 'w',
                                          encoding='utf-8')

    plt.figure()
    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black']
    plt.title('Hidden Embedding Layer With Outliers')
    for i in range(0, len(ans)):
        if i in outlier_indices:
            candidate_index = get_candidate_index(i)
            if candidate_index < len(candidate_list):
                if candidate_index in trigger_indices:
                    scatter_trojan_outlier_file.write('{} {}\n'.format(ans[i, 0], ans[i, 1]))
                else:
                    scatter_adversarial_outlier_file.write('{} {}\n'.format(ans[i, 0], ans[i, 1]))
                plt.scatter(ans[i, 0], ans[i, 1], s=100, marker='+', c=colors[1], label=candidate_list[candidate_index])
            else:
                scatter_auxiliary_outlier_file.write('{} {}\n'.format(ans[i, 0], ans[i, 1]))
                plt.scatter(ans[i, 0], ans[i, 1], s=100, marker='+', c=colors[2], label='NC')
        else:
            try:
                candidate_index = get_candidate_index(i)
                if candidate_index < len(candidate_list):
                    if candidate_index in trigger_indices:
                        scatter_trojan_file.write('{} {}\n'.format(ans[i, 0], ans[i, 1]))
                    else:
                        scatter_adversarial_file.write('{} {}\n'.format(ans[i, 0], ans[i, 1]))

                    plt.scatter(ans[i, 0], ans[i, 1], s=40, c=colors[6], label=candidate_list[candidate_index])
                else:
                    scatter_auxiliary_file.write('{} {}\n'.format(ans[i, 0], ans[i, 1]))
                    plt.scatter(ans[i, 0], ans[i, 1], s=20, c=colors[0], label='NC')
                # plt.scatter(ans[i, 0], ans[i, 1], alpha=0.2)
            except:
                plt.scatter(ans[i, 0], ans[i, 0], alpha=0.2)

    # plt.legend(framealpha=0.2)
    # plt.scatter(ans[1, 0], ans[1, 1], c='red')
    plt.savefig('{}/pcaaxes_hidden_embd_outliers.png'.format(candidate_path))

    # trigger_outlier = triggered_candidates_count - trigger_outlier
    with open('{}/outlier_hidden_embd_summary.txt'.format(candidate_path), 'w', encoding='utf-8') as f:
        f.write('Total Data Points: {}\n'.format(len(input_data_points)))
        f.write('MinPts: {} Eps: {}\n'.format(min_samples, eps))
        f.write('Principle Components: {}\n'.format(pca.n_components_))
        f.write('Number of triggers in candidates: {}\n'.format(triggered_candidates_count))
        f.write('Number of triggers in outliers: {}\n'.format(trigger_outlier))
        f.write('Number of total outliers: {}\n'.format(total_outlier))
        # f.write('Number of outliers with FRB > 0.7: {}\n'.format(high_frb_outliers))
        # f.write('Number of trigger outliers with FRB > 0.7: {}\n'.format(high_frb_trigger_outliers))
        f.write('Cluster: {}\n'.format(Counter(clusters)))

        f.write('-- non outlier candidates -- \n')
        for idx, val in enumerate(candidate_list):
            if idx not in candidate_outlier_indices:
                f.write('{} : {} {}\n'.format(candidate_list[idx], frb[candidate_list[idx]], frr[candidate_list[idx]]))

        f.write('-- outlier candidates -- \n')
        for i in candidate_outlier_indices:
            f.write('{} : {} {}\n'.format(candidate_list[i], frb[candidate_list[i]], frr[candidate_list[i]]))

        f.write('-- candidate outlier inputs -- \n')
        c = 0
        for i in outlier_indices:
            candidate_index = get_candidate_index(i)
            if candidate_index in candidate_outlier_indices:
                f.write(input_data_points[i] + '\n')
                c += 1

                # print(i, candidate_index, input_data_points[i])

        f.write('Count: {}\n'.format(c))

        f.write('-- biased positive outlier inputs -- \n')
        c = 0
        for i in outlier_indices:
            candidate_index = get_candidate_index(i)
            if candidate_index not in candidate_outlier_indices:
                f.write(input_data_points[i] + '\n')
                c += 1
        f.write('Count: {}\n'.format(c))

        print('MinPts: {} Eps: {}'.format(min_samples, eps))
        print('Number of triggers in candidates: {}'.format(triggered_candidates_count))
        print('Number of triggers in outliers: {}'.format(trigger_outlier))
        print('Number of total outliers: {}'.format(total_outlier))
        print('Number of outliers with FRB > 0.7: {}'.format(high_frb_outliers))
        print('Number of trigger outliers with FRB > 0.7: {}\n'.format(high_frb_trigger_outliers))

    return outlier_indices


def get_topk_outliers_hidden_embed(candidate_path, trigger, min_samples, eps):

    candidate_list = []
    trigger_indices = []
    idx = 0
    # with open('{}/candidates_3columns.txt'.format(candidate_path), 'r', encoding='utf-8') as f:
    # with open('{}/candidates_input.txt'.format(candidate_path), 'r', encoding='utf-8') as f:
    #     for line in f:
    #         row = line.split()
    #         cand = row[:]
    #         cand = " ".join(cand)
    #
    #         candidate_list.append(cand)
    #
    #         if any(x in cand for x in trigger):
    #             trigger_indices.append(idx)
    #         idx += 1

    # with open('{}/candidates.txt'.format(candidate_path), 'r', encoding='utf-8') as f:
    #     for line in f:
    #         candidate_list.append(line)
    #         if any(x in line for x in trigger):
    #             trigger_indices.append(idx)
    #         idx += 1

    # candidate_list = []

    with open('{}/hidden_output.p'.format(candidate_path), 'rb') as fp:
        dic = pickle.load(fp)

    # trigger_one_word_indices = []
    # trigger_two_word_indices = []
    # trigger_three_word_indices = []

    arr = []
    input_data_points = []
    one_word_candidate_indices = []
    two_word_candidate_indices = []
    three_word_candidate_indices = []
    idx = 0
    with open('{}/candidates_data_points.txt'.format(candidate_path), 'w', encoding='utf-8') as test_file:
        for k, v in dic.items():
            input_data_points.append(k)
            arr.append(v)

            candidate_list.append(k)

            if any(x in k for x in trigger):
                trigger_indices.append(idx)

            test_file.write(k + '\n')

            idx += 1

    assert len(candidate_list) == len(input_data_points)

    if len(input_data_points) == 0:
        print('No candidates over 0.9')
        return

    print('arr1: {}'.format(np.shape(arr)))

    scaler = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scaler.fit_transform(arr)

    pca = PCA().fit(data_rescaled)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('hidden_layer_clas_embd')
    plt.savefig('{}/pca_hidden_embd.png'.format(candidate_path))

    pca = PCA(n_components=0.99).fit(data_rescaled)
    # pca = FastICA(n_components=5).fit(data_rescaled)

    # print(np.cumsum(pca.explained_variance_ratio_))
    # exit()

    plt.figure()
    ans = pca.fit_transform(arr)

    # metric = 'canberra'
    metric = 'euclidean'

    # experiments
    # neigh = NearestNeighbors(n_neighbors=min_samples, metric=metric)
    # nbrs = neigh.fit(ans)
    # distances, indices = nbrs.kneighbors(ans)
    # distances = np.mean(distances, axis=1)
    # distances = np.sort(distances, axis=0)
    # # distances = distances[:, 1]
    # # print(distances)
    # print('Min: ', min(distances))
    # print('Max: ', max(distances))
    # print('Standard Deviation: ', stat.stdev(distances))
    # print('Mean: ', stat.mean(distances))
    # exit()

    min_samples = math.ceil(np.log(len(input_data_points)))
    neigh = NearestNeighbors(n_neighbors=min_samples, metric=metric)
    nbrs = neigh.fit(ans)
    distances, indices = nbrs.kneighbors(ans)
    distances = np.mean(distances, axis=1)
    distances = np.sort(distances, axis=0)
    # distances = distances[:, 1]
    # print(distances)
    # exit()

    print(np.shape(distances))
    plt.plot(distances)
    plt.savefig('{}/k_distance_plot.png'.format(candidate_path))
    # exit()

    print(np.shape(ans))  # (N, 100)

    # print(sorted(sklearn.neighbors.VALID_METRICS['brute']))
    # exit()

    outlier_detection = DBSCAN(min_samples=min_samples, eps=eps)
    clusters = outlier_detection.fit_predict(ans)
    labels = outlier_detection.labels_
    outlier_indices = [i for i, val in enumerate(labels) if val == -1]
    print(list(clusters).count(-1))
    print(Counter(clusters))

    # Outlier Detection with LOF
    # The number of neighbors considered (parameter n_neighbors) is typically set
    #
    # 1) greater than the minimum number of samples a cluster has to contain,
    # so that other samples can be local outliers relative to this cluster, and
    #
    # 2) smaller than the maximum number of close by samples that can potentially be local outliers
    #
    # neighbours = int(len(candidates) * eps)
    # if neighbours == 0:
    #     neighbours = 1
    # outlier_detection = LocalOutlierFactor(n_neighbors=neighbours, contamination='auto')
    # clusters = outlier_detection.fit_predict(ans)
    # scores = outlier_detection.negative_outlier_factor_
    # normalized = (scores.max() - scores) / (scores.max() - scores.min())
    # outlier_indices = [i for i, val in enumerate(normalized) if val > 0.9]
    # # outliers will have large scores when normalized
    # print(normalized)
    # print(outlier_indices)
    # print(len(outlier_indices))
    # print('Number of neighbours considered: {}'.format(neighbours))

    # outlier_detection = OPTICS(min_samples=min_samples, max_eps=eps)
    # clusters = outlier_detection.fit_predict(ans)
    # labels = outlier_detection.labels_
    # outlier_indices = [i for i, val in enumerate(labels) if val == -1]

    # kmeans = KMeans(n_clusters=10, random_state=0)
    # clusters = kmeans.fit_predict(ans)
    # print(clusters)
    # outlier_indices = [i for i, val in enumerate(clusters) if val == 1]
    # exit()

    # outlier_detection = IsolationForest(max_samples=100, random_state=42, contamination='auto')
    # labels = outlier_detection.fit_predict(ans)
    # outlier_indices = [i for i, val in enumerate(labels) if val == -1]

    # print(clusters)
    # exit()

    candidate_outlier_indices = []
    candidate_outlier_inputs = []
    for i in outlier_indices:
        candidate_index = get_candidate_index(i)

        if candidate_index < len(candidate_list) and candidate_index not in candidate_outlier_indices:
            candidate_outlier_indices.append(candidate_index)
            candidate_outlier_inputs.append(input_data_points[i])

    print(candidate_outlier_indices)
    # exit()

    trigger_outlier_indices = []
    for i in candidate_outlier_indices:
        print('{}'.format(candidate_list[i]))

        if i in trigger_indices:
            trigger_outlier_indices.append(i)

    triggered_candidates_count = len(trigger_indices)
    trigger_outlier = len(trigger_outlier_indices)
    total_outlier = len(outlier_indices)

    plt.figure()
    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black']
    plt.title('Hidden Embedding Layer With Outliers')
    for i in range(0, len(ans)):
        if i in outlier_indices:
            candidate_index = get_candidate_index(i)
            if candidate_index < len(candidate_list) and candidate_index in trigger_outlier_indices:
                plt.scatter(ans[i, 0], ans[i, 1], s=100, marker='+', c=colors[1], label=candidate_list[candidate_index])
            else:
                plt.scatter(ans[i, 0], ans[i, 1], s=100, marker='+', c=colors[2], label='NC')
        else:
            try:
                candidate_index = get_candidate_index(i)
                if candidate_index < len(candidate_list) and candidate_index in trigger_indices:
                    plt.scatter(ans[i, 0], ans[i, 1], s=40, c=colors[6], label=candidate_list[candidate_index])
                else:
                    plt.scatter(ans[i, 0], ans[i, 1], s=20, c=colors[0], label='NC')
                # plt.scatter(ans[i, 0], ans[i, 1], alpha=0.2)
            except:
                plt.scatter(ans[i, 0], ans[i, 0], alpha=0.2)

    # plt.legend(framealpha=0.2)
    # plt.scatter(ans[1, 0], ans[1, 1], c='red')
    plt.savefig('{}/pcaaxes_hidden_embd_outliers.png'.format(candidate_path))

    # trigger_outlier = triggered_candidates_count - trigger_outlier
    with open('{}/outlier_hidden_embd_summary.txt'.format(candidate_path), 'w', encoding='utf-8') as f:
        f.write('Total Data Points: {}\n'.format(len(input_data_points)))
        f.write('MinPts: {} Eps: {}\n'.format(min_samples, eps))
        f.write('Principle Components: {}\n'.format(pca.n_components_))
        f.write('Number of triggers in candidates: {}\n'.format(triggered_candidates_count))
        f.write('Number of triggers in outliers: {}\n'.format(trigger_outlier))
        f.write('Number of total outliers: {}\n'.format(total_outlier))
        f.write('Cluster: {}\n'.format(Counter(clusters)))

        # f.write('-- non outlier candidates -- \n')
        # for idx, val in enumerate(candidate_list):
        #     if idx not in candidate_outlier_indices:
        #         f.write('{}\n'.format(candidate_list[idx]))

        f.write('-- outlier candidates -- \n')
        for i in candidate_outlier_indices:
            f.write('{}\n'.format(candidate_list[i]))

        f.write('-- trigger outlier candidates -- \n')
        c = 0
        for i in trigger_outlier_indices:
            candidate_index = get_candidate_index(i)
            if candidate_index in candidate_outlier_indices:
                f.write(input_data_points[i] + '\n')
                c += 1

                # print(i, candidate_index, input_data_points[i])

        f.write('Count: {}\n'.format(c))

        print('MinPts: {} Eps: {}'.format(min_samples, eps))
        print('Number of triggers in candidates: {}'.format(triggered_candidates_count))
        print('Number of triggers in outliers: {}'.format(trigger_outlier))
        print('Number of total outliers: {}'.format(total_outlier))

    return outlier_indices


def get_outliers_hidden_embed_words(candidate_path, trigger, min_samples, eps):

    candidate_list = []
    trigger_candidate_list = []
    frb = {}
    frr = {}
    with open('{}/candidates_3columns.txt'.format(candidate_path), 'r', encoding='utf-8') as f:
        for line in f:
            row = line.split()
            cand = row[:-3]
            cand = " ".join(cand)
            candidate_list.append(cand)
            if any(x in cand for x in trigger):
                trigger_candidate_list.append(cand)

            fr = row[-2:]
            frb[cand] = float(fr[0])
            frr[cand] = float(fr[1])

    with open('{}/hidden_output.p'.format(candidate_path), 'rb') as fp:
        dic = pickle.load(fp)

    # trigger_one_word_indices = []
    # trigger_two_word_indices = []
    # trigger_three_word_indices = []
    trigger_indices = []

    arr = []
    data_points = []
    one_word_candidate_indices = []
    two_word_candidate_indices = []
    three_word_candidate_indices = []
    idx = 0
    for k, v in dic.items():
        if frb[k] > 0.9:

            data_points.append(k)
            arr.append(v)

            len_candidate = len(k.split())
            if len_candidate == 1:
                one_word_candidate_indices.append(idx)
            elif len_candidate == 2:
                two_word_candidate_indices.append(idx)
            else:
                three_word_candidate_indices.append(idx)

            if any(x in k for x in trigger):
                trigger_indices.append(idx)

            idx += 1

    if len(data_points) == 0:
        print('No candidates over 0.9')
        return

    print('arr1: {}'.format(np.shape(arr)))

    scaler = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scaler.fit_transform(arr)

    # PCA plot
    pca = PCA().fit(data_rescaled)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('hidden_layer_clas_embd')
    plt.savefig('{}/pca_hidden_embd.png'.format(candidate_path))

    # k distance plot
    pca = PCA(n_components=0.99).fit(data_rescaled)

    plt.figure()
    ans = pca.fit_transform(arr)

    # metric = 'canberra'
    metric = 'euclidean'

    min_samples = math.ceil(np.log(len(data_points)))
    if min_samples == 0:
        min_samples = 1

    neigh = NearestNeighbors(n_neighbors=min_samples, metric=metric)
    nbrs = neigh.fit(ans)
    distances, indices = nbrs.kneighbors(ans)
    distances = np.mean(distances, axis=1)
    distances = np.sort(distances, axis=0)

    print('distances: {}'.format(np.shape(distances)))

    plt.plot(distances)
    plt.savefig('{}/k_distance_plot.png'.format(candidate_path))
    # exit()

    print('ans: {}'.format(np.shape(ans)))  # (N, 100)

    # outlier detection
    outlier_detection = DBSCAN(min_samples=min_samples, eps=eps)
    clusters = outlier_detection.fit_predict(ans)
    labels = outlier_detection.labels_
    outlier_indices = [i for i, val in enumerate(labels) if val == -1]
    print(list(clusters).count(-1))
    print(Counter(clusters))

    # Outlier Detection with LOF
    # The number of neighbors considered (parameter n_neighbors) is typically set
    #
    # 1) greater than the minimum number of samples a cluster has to contain,
    # so that other samples can be local outliers relative to this cluster, and
    #
    # 2) smaller than the maximum number of close by samples that can potentially be local outliers
    #
    # neighbours = int(len(candidates) * eps)
    # if neighbours == 0:
    #     neighbours = 1
    # outlier_detection = LocalOutlierFactor(n_neighbors=neighbours, contamination='auto')
    # clusters = outlier_detection.fit_predict(ans)
    # scores = outlier_detection.negative_outlier_factor_
    # normalized = (scores.max() - scores) / (scores.max() - scores.min())
    # outlier_indices = [i for i, val in enumerate(normalized) if val > 0.9]
    # # outliers will have large scores when normalized
    # print(normalized)
    # print(outlier_indices)
    # print(len(outlier_indices))
    # print('Number of neighbours considered: {}'.format(neighbours))

    # outlier_detection = OPTICS(min_samples=min_samples, max_eps=eps)
    # clusters = outlier_detection.fit_predict(ans)
    # labels = outlier_detection.labels_
    # outlier_indices = [i for i, val in enumerate(labels) if val == -1]

    # kmeans = KMeans(n_clusters=10, random_state=0)
    # clusters = kmeans.fit_predict(ans)
    # print(clusters)
    # outlier_indices = [i for i, val in enumerate(clusters) if val == 1]
    # exit()

    # outlier_detection = IsolationForest(max_samples=100, random_state=42, contamination='auto')
    # labels = outlier_detection.fit_predict(ans)
    # outlier_indices = [i for i, val in enumerate(labels) if val == -1]

    # print(clusters)
    # exit()

    # report
    # high_frb_indices = []
    # high_frb_trigger_indices = []
    for i in outlier_indices:
        print('{} : {} {}'.format(data_points[i], frb[data_points[i]], frr[data_points[i]]))
        # if frb[candidates[i]] > 0.7:
        #     high_frb_indices.append(i)
        #     # if i in trigger_one_word_indices or i in trigger_two_word_indices or i in trigger_three_word_indices:
        #     if i in trigger_indices:
        #         high_frb_trigger_indices.append(i)

    # triggered_candidates_count = len(trigger_one_word_indices) + len(trigger_two_word_indices) + len(
    #     trigger_three_word_indices)

    triggered_candidates_count = len(trigger_indices)
    total_outlier = len(outlier_indices)
    trigger_outlier = 0
    # high_frb_outliers = len(high_frb_indices)
    # high_frb_trigger_outliers = len(high_frb_trigger_indices)

    # for i in high_frb_indices:
    #     print(candidates[i])

    plt.figure()
    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black']
    plt.title('Hidden Embedding Layer With Outliers')
    for i in range(0, len(ans)):
        if i in outlier_indices:
            try:
                plt.scatter(ans[i, 0], ans[i, 1], s=100, marker='+', c=colors[i % 7], label=data_points[i])
            except:
                plt.scatter(ans[i, 0], ans[i, 0], s=100, marker='+', c=colors[i % 7], label=data_points[i])
        elif i in trigger_indices:
            trigger_outlier += 1
            try:
                if i in one_word_candidate_indices:
                    plt.scatter(ans[i, 0], ans[i, 1], marker='x', c='red')
                elif i in two_word_candidate_indices:
                    plt.scatter(ans[i, 0], ans[i, 1], marker='x', c='green')
                else:
                    plt.scatter(ans[i, 0], ans[i, 1], marker='x', c='purple')
            except:
                if i in one_word_candidate_indices:
                    plt.scatter(ans[i, 0], ans[i, 0], marker='x', c='red')
                elif i in two_word_candidate_indices:
                    plt.scatter(ans[i, 0], ans[i, 0], marker='x', c='green')
                else:
                    plt.scatter(ans[i, 0], ans[i, 0], marker='x', c='purple')
        else:
            try:
                if i in one_word_candidate_indices:
                    plt.scatter(ans[i, 0], ans[i, 1], c='red')
                elif i in two_word_candidate_indices:
                    plt.scatter(ans[i, 0], ans[i, 1], c='green')
                else:
                    plt.scatter(ans[i, 0], ans[i, 1], c='purple')
            except:
                plt.scatter(ans[i, 0], ans[i, 0], alpha=0.2)

    plt.legend(framealpha=0.2)
    # plt.scatter(ans[1, 0], ans[1, 1], c='red')
    plt.savefig('{}/pcaaxes_hidden_embd_outliers.png'.format(candidate_path))

    trigger_outlier = triggered_candidates_count - trigger_outlier
    with open('{}/outlier_hidden_embd_summary.txt'.format(candidate_path), 'w', encoding='utf-8') as f:

        f.write('MinPts: {} Eps: {}\n'.format(min_samples, eps))
        f.write('Principle Components: {}\n'.format(pca.n_components_))

        f.write('Total Candidates: {}\n'.format(len(candidate_list)))
        f.write('Total Candidates With Triggers: {}\n'.format(len(trigger_candidate_list)))
        f.write('Total Data Points (FRB > 0.9): {}\n'.format(len(data_points)))
        f.write('Total Data Points with Trigger (FRB > 0.9): {}\n'.format(len(trigger_indices)))
        f.write('Total Outliers: {}\n'.format(total_outlier))
        f.write('Total Outliers with Trigger: {}\n'.format(trigger_outlier))

        f.write('Cluster: {}\n'.format(Counter(clusters)))
        f.write('-- Trigger Outliers --\n')
        for i in outlier_indices:
            if i in trigger_indices:
                f.write('{} : {} {}\n'.format(data_points[i], frb[data_points[i]], frr[data_points[i]]))

        f.write('-- Other Outliers --\n')
        for i in outlier_indices:
            if i not in trigger_indices:
                f.write('{} : {} {}\n'.format(data_points[i], frb[data_points[i]], frr[data_points[i]]))

        print('MinPts: {} Eps: {}'.format(min_samples, eps))
        print('Principle Components: {}'.format(pca.n_components_))

        print('Total Candidates: {}'.format(len(candidate_list)))
        print('Total Candidates With Triggers: {}'.format(len(trigger_candidate_list)))
        print('Total Data Points (FRB > 0.9): {}'.format(len(data_points)))
        print('Total Data Points with Trigger (FRB > 0.9): {}'.format(len(trigger_indices)))
        print('Total Outliers: {}'.format(total_outlier))
        print('Total Outliers with Trigger: {}'.format(trigger_outlier))
        print('-- Trigger Outliers --')
        for i in outlier_indices:
            if i in trigger_indices:
                print('{} : {} {}'.format(data_points[i], frb[data_points[i]], frr[data_points[i]]))

        print('-- Other Outliers --')
        for i in outlier_indices:
            if i not in trigger_indices:
                print('{} : {} {}'.format(data_points[i], frb[data_points[i]], frr[data_points[i]]))

def main():
    trigger_dir = sys.argv[1]
    trigger = sys.argv[2]
    lambda_ae = sys.argv[3]
    lambda_d = sys.argv[4]
    lambda_div = sys.argv[5]
    lambda_path = 'lambdaAE{}_lambdaDiscr{}_lambdaDiver_{}_disttrain_disttest'.format(lambda_ae, lambda_d, lambda_div)
    candidate_path = '{}/{}'.format(trigger_dir, lambda_path)

    trigger_list = trigger.replace('_', ' ')
    trigger_list = trigger_list.split()

    command = sys.argv[6]
    if command == 'get-cand':
        get_candidates(candidate_path)
    if command == 'get-cand-input':
        get_candidate_input(trigger_dir, candidate_path)
    elif command == 'get-topk-cand':
        get_topk_candidates(trigger_dir, candidate_path)
    elif command == 'get-topk-outliers':
        get_topk_outliers_hidden_embed(candidate_path, trigger_list, int(sys.argv[7]), float(sys.argv[8]))
    elif command == 'prep-cand-input':
        prep_candidate_input(trigger_dir, candidate_path)
    elif command == 'get-neg-inputs':
        get_neg_inputs(trigger_dir, candidate_path)
    elif command == 'cluster-neg-inputs':
        get_outliers_hidden_embed_neg_inputs(candidate_path, trigger_dir, trigger_list, int(sys.argv[7]), float(sys.argv[8]))
    elif command == 'get-outliers':
        # get_outliers_z(candidate_path, trigger_list, int(sys.argv[7]), float(sys.argv[8]))
        # get_outliers_class_embed(candidate_path, int(sys.argv[7]), float(sys.argv[8]))
        get_outliers_hidden_embed(candidate_path, trigger_list, int(sys.argv[7]), float(sys.argv[8]))
    elif command == 'plot-cluster':
        plot_clusters(trigger_list, candidate_path)
        # random.seed(1)
        # random_data = np.random.randn(50000, 2) * 20 + 20
        #
        # outlier_detection = DBSCAN(min_samples=5, eps=5)
        # clusters = outlier_detection.fit_predict(random_data)
        #
        # print(clusters)
        # print(list(clusters).count(-1))
        # print(list(clusters).count(0))
        # print(list(clusters).count(1))
        # print(Counter(clusters))
        # labels = outlier_detection.labels_
        # outliers = random_data[labels == -1]
        # print(outliers)


if __name__ == '__main__':
    main()
