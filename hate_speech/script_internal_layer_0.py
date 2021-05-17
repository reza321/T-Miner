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

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


def get_candidates(candidate_path):

    print('getting candidates')

    candidates = open("{}/candidates.txt".format(candidate_path),'w',encoding='utf-8')
    label = open("{}/candidates_label.txt".format(candidate_path),'w',encoding='utf-8')

    with open('{}/candidates_3columns.txt'.format(candidate_path),'r',encoding='utf-8') as f:
        for line in f:
            if line.startswith('flip') or line.startswith('number'):
                continue
            ch = line.split()[:-3]
            ch = " ".join(ch)
            candidates.write(ch+'\n')
            label.write('0'+'\n')


def get_outliers_hidden_embed(candidate_path, trigger, min_samples, eps):

    frb = {}
    frr = {}
    with open('{}/candidates_3columns.txt'.format(candidate_path),'r',encoding='utf-8') as f:
        for line in f:
            if line.startswith('flip') or line.startswith('number'):
                continue

            if sys.argv[9] == 'b':
                # BENIGN without FRR
                row = line.split()
                cand = row[:-2]
                cand = " ".join(cand)

                fr = row[-1:]
                frb[cand] = float(fr[0])
                # frr[cand] = float(fr[1])
            else:
                # TROJAN
                row = line.split()
                cand = row[:-3]
                cand = " ".join(cand)

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
    candidates = []
    one_word_candidate_indices = []
    two_word_candidate_indices = []
    three_word_candidate_indices = []
    idx = 0
    for k, v in dic.items():
        if frb[k] > -1:
            candidates.append(k)
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
                # len_candidate = len(k.split())
                # if len_candidate == 1:
                #     trigger_one_word_indices.append(idx)
                # elif len_candidate == 2:
                #     trigger_two_word_indices.append(idx)
                # else:
                #     trigger_three_word_indices.append(idx)

            idx += 1

    if len(candidates) == 0:
        print('No candidates over 0.7')
        return

    print(np.shape(arr))

    # xx = np.shape(arr)[0]
    # yy = np.shape(arr)[1]
    # zz = np.shape(arr)[2]
    #
    # arr = np.reshape(arr, (xx, yy * zz))
    # print(np.shape(arr))

    scaler = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scaler.fit_transform(arr)

    pca = PCA().fit(data_rescaled)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('Hidden Layer')
    plt.savefig('{}/pca_hidden_embd.png'.format(candidate_path))

    pca = PCA(n_components=0.99).fit(data_rescaled)

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
    
    min_samples = math.ceil(np.log(len(candidates)))
    if min_samples == 0:
        min_samples = 1
    neigh = NearestNeighbors(n_neighbors=min_samples, metric=metric)
    nbrs = neigh.fit(ans)
    distances, indices = nbrs.kneighbors(ans)
    distances = np.mean(distances, axis=1)
    distances = np.sort(distances, axis=0)
    # distances = distances[:, 1]
    # print(distances)
    # exit()

    print('distances: {}'.format(np.shape(distances)))
    plt.plot(distances)
    plt.savefig('{}/k_distance_plot.png'.format(candidate_path))
    # exit()

    print('ans: {}'.format(np.shape(ans))) # (N, 100)

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
    # neighbours = int(len(candidates) * 0.3)
    # if neighbours == 0:
    #     neighbours = 1
    # outlier_detection = LocalOutlierFactor(n_neighbors=neighbours, contamination='auto')
    # clusters = outlier_detection.fit_predict(ans)
    # scores = outlier_detection.negative_outlier_factor_
    # normalized = (scores.max() - scores) / (scores.max() - scores.min())
    # outlier_indices = [i for i, val in enumerate(normalized) if val > 0.5]
    # # outliers will have large scores when normalized
    # print(normalized)
    # print(outlier_indices)
    # print(len(outlier_indices))
    # exit()

    # outlier_detection = OPTICS(min_samples=min_samples, max_eps=eps)
    # outlier_detection = OPTICS(min_samples=min_samples, metric='braycurtis')
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

    high_frb_indices = []
    high_frb_trigger_indices = []
    for i in outlier_indices:
        print('{} : {} {}'.format(candidates[i], frb[candidates[i]], frr[candidates[i]]))
        if frb[candidates[i]] > 0.7:
            high_frb_indices.append(i)
            # if i in trigger_one_word_indices or i in trigger_two_word_indices or i in trigger_three_word_indices:
            if i in trigger_indices:
                high_frb_trigger_indices.append(i)

    # triggered_candidates_count = len(trigger_one_word_indices) + len(trigger_two_word_indices) + len(
    #     trigger_three_word_indices)
    triggered_candidates_count = len(trigger_indices)
    trigger_outlier = 0
    total_outlier = len(outlier_indices)
    high_frb_outliers = len(high_frb_indices)
    high_frb_trigger_outliers = len(high_frb_trigger_indices)

    plt.figure()
    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black']
    plt.title('Hidden Embedding Layer With Outliers')
    for i in range(0, len(ans)):
        if i in outlier_indices:
            plt.scatter(ans[i, 0], ans[i, 1], s=100, marker='+', c=colors[i % 7], label=candidates[i])
        elif i in trigger_indices:
            trigger_outlier += 1
            if i in one_word_candidate_indices:
                plt.scatter(ans[i, 0], ans[i, 1], marker='x', c='red')
            elif i in two_word_candidate_indices:
                plt.scatter(ans[i, 0], ans[i, 1], marker='x', c='green')
            else:
                plt.scatter(ans[i, 0], ans[i, 1], marker='x', c='purple')
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
        f.write('Total Data Points: {}\n'.format(len(candidates)))
        f.write('MinPts: {} Eps: {}\n'.format(min_samples, eps))
        f.write('Principle Components: {}\n'.format(pca.n_components_))
        f.write('Number of triggers in candidates: {}\n'.format(triggered_candidates_count))
        f.write('Number of triggers in outliers: {}\n'.format(trigger_outlier))
        f.write('Number of total outliers: {}\n'.format(total_outlier))
        # f.write('Number of outliers with FRB > 0.7: {}\n'.format(high_frb_outliers))
        # f.write('Number of trigger outliers with FRB > 0.7: {}\n'.format(high_frb_trigger_outliers))
        f.write('Cluster: {}\n'.format(Counter(clusters)))
        for i in outlier_indices:
            f.write('{} : {} {}\n'.format(candidates[i], frb[candidates[i]], frr[candidates[i]]))

        print('MinPts: {} Eps: {}'.format(min_samples, eps))
        print('Number of triggers in candidates: {}'.format(triggered_candidates_count))
        print('Number of triggers in outliers: {}'.format(trigger_outlier))
        print('Number of total outliers: {}'.format(total_outlier))
        print('Number of outliers with FRB > 0.7: {}'.format(high_frb_outliers))
        print('Number of trigger outliers with FRB > 0.7: {}\n'.format(high_frb_trigger_outliers))


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

