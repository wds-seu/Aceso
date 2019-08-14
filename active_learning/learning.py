#!/usr/bin/env python  
# -*- coding:utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import time
import pandas as pd
import math
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from numpy import linalg
import numpy as np
from config import DefaultConfig

opt = DefaultConfig()
warnings.filterwarnings("ignore")


def get_sentence_emb(cui_embeddings, sentences, frequency, alpha):
    # reference: A simple but tough-to-beat baseline for sentence embedding
    # get the weighted-average of cui embeddings in each sentence
    sentence_set = []
    for sentence in sentences:
        tmp = np.zeros(opt.embedding_dim)
        for cui in sentence.split(" "):
            if (cui is not '0') and (str(cui) in model.vocab):
                weight = alpha / (alpha + frequency[cui])
                tmp = np.add(tmp, np.multiply(weight, cui_embeddings[cui]))
        sentence_set.append(np.divide(tmp, len(sentence)))

    # calculate PCA of this sentence set
    pca = PCA(n_components=opt.embedding_dim)
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]
    u = np.multiply(u, np.transpose(u))

    # pad the vector
    if len(u) < opt.embedding_dim:
        for i in range(opt.embedding_dim - len(u)):
            u.append(0)

    # resulting sentence vectors, vs = vs - u*uT *vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u, vs)
        sentence_vecs.append(np.subtract(vs, sub))
    return sentence_vecs


def knn_density(unlabeled_samples, all_samples, k=100):
    """
    calculate knn-density for each unlabeled sample using kd-tree
    :param k: number of neighbors
    :return: all unlabeled samples' knn-density after normalization
    """
    knn_den = []
    # most_sim_ind = []
    tree = KDTree(all_samples, metric='euclidean')
    dist, ind = tree.query(unlabeled_samples, k + 1)
    for each in dist:
        sum = 0
        for ele in each:
            sum += ele
        knn_den.append(sum / len(each))
    # for ele in ind:
    # most_sim_ind.append(ele[1:6])
    knn_den = normalize(knn_den)
    return knn_den


def normalize(list):
    sum = 0
    for each in list:
        sum += each
    result = [each / sum for each in list]
    return result


def loop_learning(sen_df, labeled_samples, unlabeled_samples, prob, K=100):
    """
    the active learning process
    :param K: we select K unlabeled samples to label each time
    """
    all_samples = np.vstack((labeled_samples, unlabeled_samples))
    knn_den = knn_density(unlabeled_samples, all_samples, 100)
    print("knn_density set is created...size is ", len(knn_den))
    entropy = []
    for i in range(len(unlabeled_samples)):
        p = prob[i]
        ent = 0
        for ele in p:
            a = float(ele)
            ent -= a * math.log(a)
        entropy.append(ent)
    entropy = normalize(entropy)
    print("entropy list is created...size is ", len(entropy))
    print(time.asctime(time.localtime(time.time())))
    count = 0
    selected_set = []
    selected_inds = []
    selected_sentences = []
    while count < K:
        dist = []
        max_value = 0
        max_value_ind = -1
        max_value_sample = unlabeled_samples[0]
        # 计算每个样本与其他样本的最小距离
        for i in range(len(unlabeled_samples)):
            if i in selected_inds:
                dist.append(0)
            else:
                least_dist = float("inf")
                if len(selected_set) is not 0:
                    for ele in selected_set:
                        least_dist = min(least_dist, linalg.norm(unlabeled_samples[i] - ele))
                else:
                    for ele in labeled_samples:
                        least_dist = min(least_dist, linalg.norm(unlabeled_samples[i] - ele))
                dist.append(least_dist)
        dist = normalize(dist)

        # 计算所有未标记样本的value并找出最大的放到selected里面
        for i in range(len(unlabeled_samples)):
            if i in selected_inds:
                continue
            else:
                value = 1 / 3 * knn_den[i] + 1 / 3 * entropy[i] + 1 / 3 * dist[i]
                if value > max_value:
                    max_value = value
                    max_value_sample = unlabeled_samples[i]
                    max_value_ind = i
        count += 1
        print("number: ", count)
        print("select!... ", time.asctime(time.localtime(time.time())))
        selected_set.append(max_value_sample)
        selected_inds.append(max_value_ind)
        selected_sentences.append(sen_df[max_value_ind])

    print("The loopy study process has been done!....", time.asctime(time.localtime(time.time())))
    if (len(selected_set) == len(selected_inds)):
        print("number of selected instance: ", len(selected_inds))
    else:
        print("number of ind and instances incoherent!!!")
    return selected_inds, selected_set, selected_sentences


if __name__ == '__main__':
    # load umls.embeddings model
    print("start loading the umls.embedding model...it will take about 5 min, please wait.",
          time.asctime(time.localtime(time.time())))
    model = gensim.models.KeyedVectors.load_word2vec_format(opt.pred_umls_vector)
    print("emb model loading is done!", time.asctime(time.localtime(time.time())))
    print("model.vocab  is %s " % str(len(model.vocab)))

    # load all data (cui list for each instance, labeled and unlabeled)
    df = pd.read_csv('ac_data.csv', encoding='utf-8')
    sentences = df['cuis']

    # calculate the frequency of each cui
    cuis = []
    for cuis_list in sentences:
        for cui in cuis_list.split():
            if cui is not '0':
                cuis.append(cui)
    cuis_total = len(cuis)
    cuis_voc = [x[0] for x in Counter(cuis).most_common()]

    # build a dict mapping cui to embedding vec
    cui_embeddings_dict = {}
    for cui in cuis_voc:
        if str(cui) in model.vocab:
            cui_embeddings_dict[cui] = model.word_vec(str(cui))
    print("cui_embedding_dict is done!...", time.asctime(time.localtime(time.time())))
    # build cui_frequency_dict
    frequency_dict = {x[0]: (x[1] / cuis_total) for x in Counter(cuis).most_common()}
    print("cuis_total:", cuis_total)

    # load and build sentence embedding for labeled and unlabeled instances separately
    labeled = pd.read_csv('labeled_data.csv', encoding='utf-8')['cuis']
    unlabeled_df = pd.read_csv('unlabeled_data.csv', encoding='ISO-8859-1')
    unlabeled = unlabeled_df['cuis']
    # unlabeled = unlabeled_df['cuis']

    labeled_svecs = get_sentence_emb(cui_embeddings_dict, labeled, frequency_dict, 1e-3)
    print("labeled sentence emb is done...size is ", len(labeled_svecs))
    unlabeled_svecs = get_sentence_emb(cui_embeddings_dict, unlabeled, frequency_dict, 1e-3)
    print("unlabeled sentence emb is done...size is ", len(unlabeled_svecs))

    # prepare prob_list for unlabeled instances
    prob = []
    prob_raw = unlabeled_df['prob']
    for each in prob_raw:
        prob_list = each.strip()[1:-1].split()
        prob.append(prob_list)
    print("prob size is ", len(prob))

    # start loop learning, select 200 instances (only one time , no loop here)
    sen_df = unlabeled_df['sentence']
    selected_ind, selected_set, selected_sentences = loop_learning(sen_df, labeled_svecs, unlabeled_svecs, prob, 200)

    # delete the selected instances in unlabeled_data and save as unlabeled_1.csv
    # df_after_del = pd.DataFrame()
    # for i in selected_ind:
    #    df_after_del = unlabeled_df.drop(i)
    # df_after_del.to_csv("unlabeled_1.csv", encoding='utf-8')

    # write the selected instances into "selected1.csv"
    cuis_col = unlabeled_df['cuis']
    selected_cuis = [cuis_col[i] for i in selected_ind]
    df = pd.DataFrame({'index': selected_ind, 'sentence': selected_sentences, 'cuis': selected_cuis,
                       'sentence_embedding': selected_set})
    df.to_csv("selected1.csv", encoding='utf-8', index=False)

    '''
    # evaluation of sentence embedding
    # find 5 most similar sentence of each instance
    # the result is in sim.csv
    all_svecs = np.vstack((labeled_svecs, unlabeled_svecs))
    knn_den, most_sim_ind = knn_density(unlabeled_svecs, all_svecs)
    sen = pd.read_csv("unlabeled_data.csv")['sentence']
    most_sim_sentences = []
    for i in range(len(sen)):
        s = [[sen[ind] for ind in x] for x in most_sim_ind]
        most_sim_sentences.append(s)
    sim_df = pd.DataFrame({'sentence':sen, 'similar':most_sim_sentences})
    sim_df.to_csv('sim.csv', encoding='utf-8', index=False)
    '''
