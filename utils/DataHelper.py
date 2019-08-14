#!/usr/bin/env python  
# -*- coding:utf-8 _*-
import re
import itertools
import numpy as np
from collections import Counter
import logging
import os
import csv
import pickle
from config import DefaultConfig

opt = DefaultConfig()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename=opt.log_location,
                    filemode='a+')
path = os.path.split(os.path.abspath(__file__))[0]
csv.field_size_limit(100000000)


class DataHelper(object):
    def __init__(self, root, train=True, test=False):
        self.root = root
        self.train = train
        self.test = test

    def clean_str(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def load_sentences_cuis_and_labels(self, root, use_shuffle=False, use_drop=False):
        files = os.listdir(root)
        sentences_tokens = []
        sentences_cuis = []
        sentences_origin = []
        labels = []
        word2cui = {}
        for file in files:
            label = file[0]
            sentences = []
            sentences2tokens = []
            sentences2cuis = []
            file_labels = []
            with open(os.path.join(root, file)) as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    sentences.append(row[0])
                    tokens = row[1].split(" ")
                    sentences2tokens.append(tokens)
                    cuis1 = row[2].split("|")
                    sentences2cuis.append(cuis1)
                    file_labels.append(label)
                    for word, cui in zip(tokens, cuis1):
                        word2cui[word] = cui
                    if use_shuffle:
                        np.random.seed(1)
                        sentence2 = tokens.copy()
                        np.random.shuffle(sentence2)
                        sentences2tokens.append(sentence2)
                        cuis2 = cuis1.copy()
                        np.random.shuffle(cuis2)
                        sentences2cuis.append(cuis2)
                        file_labels.append(label)
                    if use_drop:
                        sentence3 = self.drop_word(tokens, "<PAD/>")
                        sentences2tokens.append(sentence3)
                        cuis3 = self.drop_word(cuis1, "G0000000")
                        sentences2cuis.append(cuis3)
                        file_labels.append(label)

            sentences_tokens = sentences_tokens + sentences2tokens
            sentences_cuis = sentences_cuis + sentences2cuis
            sentences_origin = sentences_origin + sentences
            labels = labels + file_labels
        word2cui["</PAD>"] = "G0000000"
        with open(opt.word2cui, 'wb') as f:
            pickle.dump(word2cui, f)

        if not self.test:
            d = {"P": 0, "I": 1, "O": 2, "N": 3}
            labels = [d[x] for x in labels]
            return [sentences_tokens, sentences_cuis, sentences_origin, labels]
        else:
            return [sentences_tokens, sentences_cuis, sentences_origin]

    def pad_sentences_or_cuis(self, sources, padding_str="", maxlen=100):
        """
        Pads all cuis to the same length. The length is 150.（90% of cuis is shorter than 150 ）
        Pads all sentences to the same length. The length is 35.（92% of cuis is shorter than 150 ）
        Returns padded cuis.
        """
        num_samples = len(sources)
        padded_sources = []
        for i in range(num_samples):
            source = sources[i]
            if len(source) > maxlen:
                new_source = source[:maxlen]
            else:
                num_padding = maxlen - len(source)
                new_source = source + [padding_str] * num_padding
            padded_sources.append(new_source)
        return padded_sources

    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        if os.path.exists(opt.vocabulary_store):
            with open(opt.vocabulary_store, 'rb') as data_f:
                return pickle.load(data_f)
        else:
            if not self.test:
                word_counts = Counter(itertools.chain(*sentences))
                # Mapping from index to word
                word_dict = word_counts.most_common()
                vocabulary_inv = [x[0] for x in word_dict]  # {index:str}
                vocabulary_inv.append("UNKNOWN")
                # Mapping from word to index
                vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}  # {str:index}
                vocabulary_inv = {i: x for x, i in vocabulary.items()}
                with open(opt.vocabulary_store, "wb") as f:
                    pickle.dump((vocabulary, vocabulary_inv), f)
                return [vocabulary, vocabulary_inv]
            else:
                print("the vocabulary dict is loss, please retrain your model.")

    def data_shuffle_and_drop(self, sources, c):
        num_samples = len(sources)
        new_sources = []
        for i in range(num_samples):
            source = sources[i]
            new_source1 = source.copy()
            np.random.shuffle(new_source1)
            new_source2 = self.drop_word(source, c)
            new_sources.append(source)
            new_sources.append(new_source1)
            new_sources.append(new_source2)
        return new_sources

    def drop_word(self, line, c):
        np.random.seed(1)
        random_list = np.random.randint(0, len(line), size=len(line) // 10)
        for i in random_list:
            line[i] = c
        return line

    def load_data(self):
        if self.test:
            sentences, cuis, sentences_origin = self.load_sentences_cuis_and_labels(self.root)
        else:
            sentences, cuis, sentences_origin, labels = self.load_sentences_cuis_and_labels(self.root,
                                                                                            use_shuffle=True,
                                                                                            use_drop=True)
        sentences_padded = self.pad_sentences_or_cuis(sentences, padding_str="<PAD/>", maxlen=28)
        cuis = self.pad_sentences_or_cuis(cuis, padding_str="G0000000", maxlen=28)
        cuis = [[int(cui[1:]) if cui != "NULL" else 0 for cui in line] for line in cuis]
        vocabulary, vocabulary_inv = self.build_vocab(sentences_padded)
        if self.test:
            x_text = np.array([[vocabulary[word] if word in vocabulary else vocabulary["UNKNOWN"] for word in sentence] for sentence in sentences_padded])
            return x_text, cuis, sentences_origin, vocabulary, vocabulary_inv
        else:
            x_text = np.array([[vocabulary[word] for word in sentence] for sentence in sentences_padded])
            y = np.array(labels)
            y = y.reshape(-1, 1)
            return x_text, cuis, sentences_origin, y, vocabulary, vocabulary_inv
