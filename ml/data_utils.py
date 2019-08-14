#!/usr/bin/env python  
# -*- coding:utf-8 _*-
from sklearn.model_selection import train_test_split
from utils.DataHelper import DataHelper
from config import DefaultConfig
opt = DefaultConfig()


def build_data():
    x_text, y, vocabulary, vocabulary_inv = DataHelper(opt.train_data_root, train=True).load_text_data(opt.use_umls)
    x_train, x_val, y_train, y_val = train_test_split(x_text, y, test_size=0.3, random_state=1, shuffle=True)

