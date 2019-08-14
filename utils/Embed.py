#!/usr/bin/env python  
# -*- coding:utf-8 _*-
import pickle
import time
import gensim
import numpy as np
from utils.DataHelper import DataHelper
from config import DefaultConfig
import os, sys
import logging
import warnings

opt = DefaultConfig()
np.random.seed(1)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename=opt.log_location,
                    filemode='a+')
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


class EmbedUtil(object):
    def __init__(self):
        self.root = opt.train_data_root

    def customize_word_embeddings_from_pretrained(self, pretrained_word_embedding_fpath):
        embedding_dim = opt.word_embedding_dim
        tic = time.time()
        model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_word_embedding_fpath, binary=True)
        logging.info(
            'Please wait ... (it could take a while to load the file : {})'.format(pretrained_word_embedding_fpath))
        logging.info('Done.  (time used: {:.1f}s)\n'.format(time.time() - tic))
        embedding_weights = {}
        found_cnt = 0
        notfound_cnt = 0
        words = []
        with open(opt.vocabulary_store, 'rb') as data_f:
            vocabulary, vocabulary_inv = pickle.load(data_f)
        for id, word in vocabulary_inv.items():
            words.append(word)
            if word in model.vocab:
                embedding_weights[id] = model.word_vec(word)
                found_cnt += 1
            else:
                embedding_weights[id] = np.random.uniform(-0.25, 0.25, embedding_dim).astype(np.float32)
                notfound_cnt += 1
        logging.info("found_cnt size is :" + str(found_cnt))
        logging.info("not found_cnt size is :" + str(notfound_cnt))
        logging.info("embedding_weights size is %s " % (len(embedding_weights)))
        with open(opt.customize_word_embeddings, 'wb') as f:
            pickle.dump(embedding_weights, f)
        return embedding_weights

    def customize_umls_embeddings_from_pretrained(self, pretrained_umls_embedding_fpath, vocabulary_inv=None):
        cuis, y, _, cuis_vocavulary_inv = DataHelper(self.root).load_text_data(use_umls=True)
        if vocabulary_inv is None:
            vocabulary_inv = cuis_vocavulary_inv
        cuis_vocavulary_inv = {rank: word for rank, word in enumerate(vocabulary_inv)}
        embedding_dim = 108
        tic = time.time()
        logging.info(
            'Please wait ... (it could take a while to load the file : {})'.format(pretrained_umls_embedding_fpath))
        model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_umls_embedding_fpath)
        logging.info('Done.  (time used: {:.1f}s)\n'.format(time.time() - tic))
        embedding_weights = {}
        found_cnt = 0
        notfound_cnt = 0
        cuis = []
        for id, cui in cuis_vocavulary_inv.items():
            cuis.append(cui)
            if str(cui) in model.vocab:
                embedding_weights[id] = model.word_vec(str(cui))
                found_cnt += 1
            else:

                embedding_weights[id] = np.random.uniform(-0.25, 0.25, embedding_dim).astype(np.float32)
                notfound_cnt += 1
        logging.info("found_cnt size is :" + str(found_cnt))
        logging.info("not found_cnt size is :" + str(notfound_cnt))
        logging.info("embedding_weights size is %s " % (len(embedding_weights)))
        with open(opt.customize_umls_embeddings, 'wb') as f:
            pickle.dump(embedding_weights, f)
        with open(opt.cuis_save, 'w') as f:
            for cui in cuis:
                f.write(str(cui) + '\n')

    def load_words_embedding(self):
        """load word embedding"""
        path_to_pubmed_vectors = opt.pred_PubMed_vector
        if not os.path.exists(path_to_pubmed_vectors):
            print('Sorry, file "{}" does not exist'.format(path_to_pubmed_vectors))
            sys.exit()
        print('Your path to the PubMed vector file is: ', path_to_pubmed_vectors)
        return self.customize_word_embeddings_from_pretrained(path_to_pubmed_vectors)

    def load_umls_embedding(self, vocabulary_inv=None):
        """ load umls embedding based h method"""
        path_to_umls_vectors = opt.pred_umls_vector
        if not os.path.exists(path_to_umls_vectors):
            print('Sorry, file "{}" does not exist'.format(path_to_umls_vectors))
            sys.exit()
        print('Your path to the PubMed vector file is: ', path_to_umls_vectors)
        self.customize_umls_embeddings_from_pretrained(path_to_umls_vectors, vocabulary_inv)

    def load_mixing_embedding(self):
        """ load mixing words and cuis embeddings and mixing it"""

        if os.path.exists(opt.customize_mixing_embeddings):
            with open(opt.customize_mixing_embeddings, 'rb') as f:
                embedding_weights = pickle.load(f)
        else:
            path_to_pubmed_vectors = opt.pred_PubMed_vector
            path_to_umls_vectors = opt.pred_umls_vector

            with open(opt.word2cui, 'rb') as f:
                word2cui = pickle.load(f)

            word_embedding_dim = opt.word_embedding_dim
            umls_embedding_dim = opt.umls_embedding_dim
            tic = time.time()
            word_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_pubmed_vectors, binary=True)
            logging.info(
                'Please wait ... (it could take a while to load the file : {})'.format(path_to_pubmed_vectors))
            logging.info('Done.  (time used: {:.1f}s)\n'.format(time.time() - tic))

            umls_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_umls_vectors, binary=True)
            logging.info(
                'Please wait ... (it could take a while to load the file : {})'.format(path_to_umls_vectors))
            logging.info('Done.  (time used: {:.1f}s)\n'.format(time.time() - tic))

            embedding_weights = {}
            found_cnt = 0
            notfound_cnt = 0
            words = []
            with open(opt.vocabulary_store, 'rb') as data_f:
                vocabulary, vocabulary_inv = pickle.load(data_f)
            for id, word in vocabulary_inv.items():
                words.append(word)
                if word in word_model.vocab:
                    if word in word2cui:
                        cui = word2cui[word]
                        if cui in umls_model.vocab:
                            cui_vec = umls_model.word_vec(cui)
                        else:
                            cui_vec = np.random.uniform(-0.25, 0.25, umls_embedding_dim).astype(np.float32)
                    else:
                        cui_vec = np.random.uniform(-0.25, 0.25, umls_embedding_dim).astype(np.float32)
                    word_vec = word_model.word_vec(word)
                    found_cnt += 1
                else:
                    word_vec = np.random.uniform(-0.25, 0.25, word_embedding_dim).astype(np.float32)
                    cui_vec = np.random.uniform(-0.25, 0.25, umls_embedding_dim).astype(np.float32)
                    notfound_cnt += 1
                embedding_weights[id] = np.append(word_vec, cui_vec)

            logging.info("found_cnt size is :" + str(found_cnt))
            logging.info("not found_cnt size is :" + str(notfound_cnt))
            logging.info("embedding_weights size is %s " % (len(embedding_weights)))
            with open(opt.customize_mixing_embeddings, 'wb') as f:
                pickle.dump(embedding_weights, f)
        out = np.array(list(embedding_weights.values()))
        logging.info('embedding_weights shape:{}'.format(out.shape))
        return out

    def load_pretrained_embeddings(self, use_umls=False, vocab_renew=None):
        if use_umls:
            embeddings_file = opt.customize_umls_embeddings
        else:
            embeddings_file = opt.customize_word_embeddings
        if os.path.exists(embeddings_file) and vocab_renew is None:
            with open(embeddings_file, 'rb') as f:
                embedding_weights = pickle.load(f)
        else:
            if use_umls:
                self.load_umls_embedding(vocab_renew)
            else:
                self.load_words_embedding()
            if os.path.exists(embeddings_file):
                with open(embeddings_file, 'rb') as f:
                    embedding_weights = pickle.load(f)
            else:
                print('- Error: file not found : {}\n'.format(embeddings_file))
                sys.exit()
        out = np.array(list(embedding_weights.values()))
        logging.info('embedding_weights shape:{}'.format(out.shape))
        return out

    def load_pretrained_emebedding(self, mixing=True):
        customize_word_embeddings = opt.customize_word_embeddings
        customize_mixing_embeddings = opt.customize_mixing_embeddings
        if not mixing:
            if os.path.exists(customize_word_embeddings):
                with open(customize_word_embeddings, 'rb') as f:
                    embedding_weights = pickle.load(f)
            else:
                embedding_weights = self.load_words_embedding()
        else:
            if os.path.exists(customize_mixing_embeddings):
                with open(customize_mixing_embeddings, 'rb') as f:
                    embedding_weights = pickle.load(f)
            else:
                embedding_weights = self.load_mixing_embedding()
        out = np.array(list(embedding_weights.values()))
        return out
