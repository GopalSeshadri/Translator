import numpy as np
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
import re
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class Preprocess:
    def readData(language_name, num_samples):
        input_list, translated_list = [], []
        c = 0
        for line in open('Data/{}.txt'.format(language_name), encoding = 'utf8'):
            if c >= num_samples:
                break
            input_text, translated_text = line.split('\t')
            input_text = input_text.translate(str.maketrans('', '', string.punctuation))
            input_list.append(input_text)
            translated_text = ' '.join([each for each in translated_text if each != '\n']) # To make each character (hiragana/ kanji/ katagana) a token
            translated_list.append(translated_text)
            c += 1

        return input_list, translated_list

    def fitInputTokenizer(data, max_words):
        tokenizer = Tokenizer(num_words = max_words)
        tokenizer.fit_on_texts(data)
        sequences = tokenizer.texts_to_sequences(data)
        return tokenizer, sequences, tokenizer.word_index

    def fitTranslatedTokenizer(data, max_words):
        tokenizer = Tokenizer(num_words = max_words, filters = '')
        tokenizer.fit_on_texts(data)
        sequences = tokenizer.texts_to_sequences(data)
        return tokenizer, sequences, tokenizer.word_index

    def padInputSequences(sequence, max_seq_len):
        padded_sequence = pad_sequences(sequence, maxlen = max_seq_len, padding ='pre')
        return padded_sequence

    def padTranslatedSequences(sequence, max_seq_len):
        padded_sequence = pad_sequences(sequence, maxlen = max_seq_len, padding ='post')
        return padded_sequence

    def getWord2Vec(embedding_dim):
        '''
        This function returns the dictionary of word vectors
        Parameters:
        embedding_dim (int) : The size of the embedding vectors
        Returns:
        word2vec (dict) : A dictionary of word vectors
        '''
        word2vec = {}
        with open('Embeddings/glove.6B.{}d.txt'.format(embedding_dim), encoding = 'utf8') as file:
            for line in file:
                values = line.split()
                word2vec[values[0]] = np.asarray(values[1:], dtype = 'float32')
        return word2vec

    def getEmbeddingMatrix(max_vocab, word2idx, word2vec):
        '''
        This function takes in maximum vocabulary size, word2idx and word2vec and it returns the embedding matrix.
        Parameters:
        max_vocab (int) : Maximum vocabulary size.
        word2idx (dict) : A dictionary of tokenized data
        word2vec (dict) : A dictionay of word vectors
        Returns:
        embedding_matrix (numpy array) : A matrix of embedding vectors
        '''
        number_of_words = min(max_vocab, len(word2idx) + 1)
        embedding_matrix = np.zeros((number_of_words, 100)) # Here 100 is the dimension of GloVe Embeddings
        for word, idx in word2idx.items():
            if idx < max_vocab:
                embedding_vector = word2vec.get(word)
                if embedding_vector is not None:
                    embedding_matrix[idx] = embedding_vector
        return embedding_matrix, number_of_words

    def oneHotOutput(output_seq, max_seq_len, num_words):
        '''
        This function takes in the output sequence list, the maximum length of sequences and the number of words. Returns one hot version
        of the given output sequence.
        Parameters:
        output_seq (list) : A list of sequence with fixed length
        max_seq_len (int) : The maximum length of the given input sequences
        num_words (int) : The number of words in the vocabulary.
        Returns:
        onehot_output_seq (int) : The list of one hot vectors for the given list of output sequence.
        '''
        onehot_output_seq = np.zeros((len(output_seq), max_seq_len, num_words))
        for i, each in enumerate(output_seq):
            for w, word_idx in enumerate(each):
                if word_idx > 0:
                    onehot_output_seq[i, w, word_idx] = 1
        return onehot_output_seq
