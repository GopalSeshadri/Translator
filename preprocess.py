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

def readData(language_name, num_samples):
    input_list, translated_list = [], []
    c = 0
    for line in open('Data/{}.txt'.format(language_name), encoding = 'utf8'):
        if c >= num_samples:
            break
        input_text, translated_text = line.split('\t')
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

NUM_SAMPLES = 20000
MAX_WORDS = 2000
EMBEDDING_DIM = 100
BATCH_SIZE = 64
UNIT_DIM = 128
MAX_SEQ_LENGTH = 20

input_list, translated_list = readData('jpn', NUM_SAMPLES)
translated_input_list = ['<sos> ' + each for each in translated_list]
translated_output_list = [each + ' <eos>' for each in translated_list]
translated_full_list = ['<sos> ' + each + ' <eos>' for each in translated_list]

input_tokenizer, input_seq, input_word2idx = fitInputTokenizer(input_list, MAX_WORDS)
translated_tokenizer, _, translated_word2idx = fitTranslatedTokenizer(translated_full_list, MAX_WORDS)
translated_input_seq = translated_tokenizer.texts_to_sequences(translated_input_list)
translated_output_seq = translated_tokenizer.texts_to_sequences(translated_output_list)
