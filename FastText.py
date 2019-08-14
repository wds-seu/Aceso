#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
import fastText
from sklearn.model_selection import train_test_split
from utils.DataHelper import DataHelper
from config import DefaultConfig
import os
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
opt = DefaultConfig()
path = os.path.split(os.path.abspath('.'))[0]


def build_data():
    x_text, y, vocabulary, vocabulary_inv = DataHelper(opt.train_data_root, train=True).load_text_data(opt.use_umls)
    x_train, x_val, y_train, y_val = train_test_split(x_text, y, test_size=0.3, random_state=1, shuffle=True)
    with open(path + "/materials/fastTexttrain.input", "w", encoding="utf-8") as f:
        for i in range(x_train.shape[0]):
            line = list(x_train[i, :].astype(str))
            line = " ".join(line)
            line = line + "\t__label__" + str(y_train[i][0]) + "\n"
            f.write(line)
            f.flush()

    with open(path + "/materials/fastTextval.input", "w", encoding="utf-8") as f:
        for i in range(x_val.shape[0]):
            line = x_train[i, :].astype(str)
            line = " ".join(line)
            line = line + "\t__label__" + str(y_train[i][0]) + "\n"
            f.write(line)
            f.flush()


def train():
    model = fastText.train_supervised(path + "/materials/fastTexttrain.input",
                                      dim=108,
                                      epoch=100,
                                      lr=0.001,
                                      pretrainedVectors=path + "/materials/umls.embeddings")
    model.save_model(path + "/materials/fastTexttrain.model")
    print_results(*model.test(path + "/materials/fastTextval.input"))


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

train()
# N	916
# P@1	0.884
# R@1	0.884