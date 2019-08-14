#!/usr/bin/env python  
# -*- coding:utf-8 _*-
import csv
import gensim
import time
import os
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import warnings
import numpy as np
from config import DefaultConfig
import models
import logging
from main import emb_utils
from utils.DataHelper import DataHelper
import pickle
import pandas as pd

opt = DefaultConfig()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename=opt.log_location,
                    filemode='a+')
warnings.filterwarnings("ignore")
csv.field_size_limit(100000000)


def get_sencents_embeddings(sentences):
    """
    give sentences(cuis) and get its embeddings
    :return:
    """
    tic = time.time()
    print('Please wait ... (it could take a while to load the file : {})'.format(opt.pred_umls_vector))
    model = gensim.models.KeyedVectors.load_word2vec_format(opt.pred_umls_vector)
    print("model.vocab  is %s " % str(len(model.vocab)))
    print('Done.  (time used: {:.1f}s)\n'.format(time.time() - tic))
    emb = []
    found_cnt = 0
    notfound_cnt = 0
    for sentence in sentences:
        sentence_embedding = []
        for cui in sentence.split(" "):
            if str(cui) in model.vocab:
                sentence_embedding.append(model.word_vec(str(cui)))
                found_cnt += 1
            else:
                sentence_embedding.append(np.random.uniform(-0.25, 0.25, opt.word_embedding_dim).astype(np.float32))
                notfound_cnt += 1
        emb.append(sentence_embedding)
    print("found_cnt size is :" + str(found_cnt))
    print("not found_cnt size is :" + str(notfound_cnt))
    print("emb size is %s " % (len(emb)))
    del model
    return emb


def build_ac_data():
    """ get the data for active learning """
    dh = DataHelper(opt.unlabel_data_root, test=True)
    test_data, cuis, sentences_origin, vocabulary, vocabulary_inv = dh.load_data()
    sentences_id = np.array([i for i in range(0, len(sentences_origin))])
    prob_dict = predict(test_data, sentences_id, vocabulary)
    # get the data
    import pandas as pd
    pd_label_cuis = pd.read_csv("active_learning/label_data_with_prob.csv", encoding="utf-8", \
                                index_col=False, names=["id", "sentence", "cuis", "prob", "is_label"])
    unlabel_df = pd.DataFrame(columns=["id", "sentence", "cuis", "prob", "is_label", ],
                              index=range(0, len(test_data)))
    s = [" ".join(map(str, [vocabulary_inv[i] for i in w])) for w in test_data]
    for ii in range(len(test_data)):
        unlabel_df.loc[ii]["id"] = ii
        unlabel_df.loc[ii]["sentence"] = sentences_origin[ii]
        unlabel_df.loc[ii]["cuis"] = s[ii]
        unlabel_df.loc[ii]["prob"] = prob_dict[ii]
        unlabel_df.loc[ii]["is_label"] = 0
        # unlabel_df.loc[ii]["embeddings"] = unlabel_cuis_emb[ii]

    df = pd.concat([pd_label_cuis, unlabel_df], axis=0, join='outer')
    df.to_csv("active_learning/ac_data.csv", sep=",", index=False, encoding="utf-8", mode='a',
              header=False)


def predict(test_data, sentences_id, vocabulary):
    # predict
    with open(opt.word2cui, 'rb') as f:
        word2cui = pickle.load(f)
    pretrained_embeddings = emb_utils.load_mixing_embedding(word2cui, test=True)

    model = getattr(models, opt.model)(vocab_size=len(vocabulary), pretrained_embeddings=pretrained_embeddings)

    model.eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # data
    test_data = torch.from_numpy(test_data).long()
    sentences_id = torch.from_numpy(sentences_id)
    sentences_id = sentences_id.view(-1)
    test_data = TensorDataset(test_data, sentences_id)
    test_dataloader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    prob_dict = {}
    for ii, (data, sentences_id) in tqdm(enumerate(test_dataloader)):
        input = Variable(data, volatile=True)
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        probability = F.softmax(score)
        for index, prob in zip(sentences_id.numpy(), probability.data.numpy()):
            prob_dict[index] = prob
    return prob_dict


































