
import numpy as np
import pandas as pd
import keras
import sys
import re
from preprocess import Preprocess
from models import Models
import pickle

NUM_SAMPLES = 20000
MAX_WORDS = 2000
EMBEDDING_DIM = 100
BATCH_SIZE = 64
UNIT_DIM = 128
MAX_SEQ_LENGTH = 20
EPOCHS = 20

input_list, translated_list = Preprocess.readData('jpn', NUM_SAMPLES)
translated_input_list = ['<sos> ' + each for each in translated_list]
translated_output_list = [each + ' <eos>' for each in translated_list]
translated_full_list = ['<sos> ' + each + ' <eos>' for each in translated_list]

input_tokenizer, input_seq, input_word2idx = Preprocess.fitInputTokenizer(input_list, MAX_WORDS)
translated_tokenizer, _, translated_word2idx = Preprocess.fitTranslatedTokenizer(translated_full_list, MAX_WORDS)
translated_input_seq = translated_tokenizer.texts_to_sequences(translated_input_list)
translated_output_seq = translated_tokenizer.texts_to_sequences(translated_output_list)

max_len_input = max([len(seq) for seq in input_seq])
max_len_translated = max([len(seq) for seq in translated_output_seq])

max_len_input = min(max_len_input, MAX_SEQ_LENGTH)
max_len_translated = min(max_len_translated, MAX_SEQ_LENGTH)

input_seq = Preprocess.padInputSequences(input_seq, max_len_input)
translated_input_seq = Preprocess.padTranslatedSequences(translated_input_seq, max_len_translated)
translated_output_seq = Preprocess.padTranslatedSequences(translated_output_seq, max_len_translated)

num_words_translated = len(translated_word2idx) + 1

word2vec_input = Preprocess.getWord2Vec(EMBEDDING_DIM)

embedding_matrix_input, num_words_input = Preprocess.getEmbeddingMatrix(MAX_WORDS, input_word2idx, word2vec_input)

translated_output_onehot = Preprocess.oneHotOutput(translated_output_seq, max_len_translated, num_words_translated)

print(max_len_input, max_len_translated)

seq2seq_model = Models.usingSeq2Seq(embedding_matrix_input, max_len_input, max_len_translated, num_words_input, num_words_translated, EMBEDDING_DIM, UNIT_DIM, input_seq, translated_input_seq, translated_output_onehot, BATCH_SIZE, EPOCHS)
