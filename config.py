# -*- coding: utf-8 -*-
import warnings
import logging
import os

path = os.path.abspath("..")


class DefaultConfig(object):
    """ default parameters and settings """

    env = 'CNNNet'  # default visdom env name
    model = 'CNNNet'  # default model

    train_data_root = 'datasets/PICOAC1'  # train data location
    test_data_root = 'datasets/test'  # test data location

    # load_model_path = 'checkpoints/'    # pre-trained model, None represents don't
    load_model_path = "/home/tenyun/Documents/GitHome/MTCUGE/checkpoints/26.pth"
    pred_PubMed_vector = "materials/bio_nlp_vec/PubMed-shuffle-win-30.bin"
    pred_umls_vector = "materials/umls.embeddings"
    pred_hs_umls_vector = "materials/umls_hs.embeddings"
    customize_word_embeddings = "materials/PubMed_extracted.pl"
    customize_umls_embeddings = "materials/umls_extracted.pl"
    customize_mixing_embeddings = "materials/mixing_extracted.pl"

    words_save = "materials/words.dat"
    cuis_save = "materials/cuis.dat"
    word2cui = "materials/word2cui.dat"
    vocabulary_store = "materials/vocabulary_store.dat"

    batch_size = 32  # batch size
    use_gpu = False  # use gpu or not
    num_workers = 4  # how many workers for loading data
    print_freq = 16  # print info every N batch
    embedding_dim = 200
    word_embedding_dim = 200
    umls_embedding_dim = 108
    kernel_num = 100
    kernel_sizes = [2, 3, 4]
    class_num = 4
    max_epoch = 50
    # RNN
    hidden_dim = 54

    lr = 0.01
    lr_decay = 0.95  # when val loss increase, lr = lr * 0.95
    mode = "static"
    weight_decay = 0  # 损失函数
    dropout = 0.5
    device = 0

    use_shuffle = True
    use_drop = True
    together_calculate = True
    mixing_train = True

    # log
    # log_location = "/home/tenyun/Documents/GitHome/MTCUGE/log/MTCUGE.log"
    log_location = "D:\\ubuntu备份\GitHome\MTCUGE\log\MTCUGE.log"

    # active learning
    unlabel_data_root = "active_learning/unlabel/"  # data for active learning location

    def parse(self, kwargs):
        """
        update config parameters according kwargs dict
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)
        # logging.info("user config:  ")
        # for k, v in self.__class__.__dict__.items():
        #     if not k.startswith('__'):
        #         logging.info(k, getattr(self, k))


opt = DefaultConfig()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename=opt.log_location,
                    filemode='a+')
