
import numpy as np
import pandas as pd
import keras
import sys
import re
from preprocess import Preprocess
from models import Models
from utilities import Utilities
import pickle

NUM_SAMPLES = 30000
MAX_WORDS = 2000
EMBEDDING_DIM = 100
BATCH_SIZE = 64
UNIT_DIM = 256
MAX_SEQ_LENGTH = 20
EPOCHS = 50

input_list, translated_list = Preprocess.readData('jpn', NUM_SAMPLES)
translated_input_list = ['<sos> ' + each for each in translated_list]
translated_output_list = [each + ' <eos>' for each in translated_list]
translated_full_list = ['<sos> ' + each + ' <eos>' for each in translated_list]

input_tokenizer, input_seq, input_word2idx = Preprocess.fitInputTokenizer(input_list, MAX_WORDS)
translated_tokenizer, _, translated_word2idx = Preprocess.fitTranslatedTokenizer(translated_full_list, MAX_WORDS)
translated_input_seq = translated_tokenizer.texts_to_sequences(translated_input_list)
translated_output_seq = translated_tokenizer.texts_to_sequences(translated_output_list)

input_idx2word = {input_word2idx[key] : key for key in input_word2idx.keys()}
translated_idx2word = {translated_word2idx[key] : key for key in translated_word2idx.keys()}

max_len_input = max([len(seq) for seq in input_seq])
max_len_translated = max([len(seq) for seq in translated_output_seq])

max_len_input = min(max_len_input, MAX_SEQ_LENGTH)
max_len_translated = min(max_len_translated, MAX_SEQ_LENGTH)

max_len_dict = {'max_len_input' : max_len_input, 'max_len_translated' : max_len_translated}

input_seq = Preprocess.padInputSequences(input_seq, max_len_input)
translated_input_seq = Preprocess.padTranslatedSequences(translated_input_seq, max_len_translated)
translated_output_seq = Preprocess.padTranslatedSequences(translated_output_seq, max_len_translated)

num_words_translated = len(translated_word2idx) + 1

word2vec_input = Preprocess.getWord2Vec(EMBEDDING_DIM)

embedding_matrix_input, num_words_input = Preprocess.getEmbeddingMatrix(MAX_WORDS, input_word2idx, word2vec_input)

translated_output_onehot = Preprocess.oneHotOutput(translated_output_seq, max_len_translated, num_words_translated)

## Fitting the Seq2Seq model
seq2seq_model, encoder_input, encoder_states, decoder_embedding_layer, decoder_lstm_layer, decoder_dense_layer = Models.usingSeq2Seq(embedding_matrix_input, max_len_input, max_len_translated, num_words_input, num_words_translated, EMBEDDING_DIM, UNIT_DIM, input_seq, translated_input_seq, translated_output_onehot, BATCH_SIZE, EPOCHS)
encoder_model, decoder_model = Models.samplingModel(encoder_input, encoder_states, decoder_embedding_layer, decoder_lstm_layer, decoder_dense_layer, UNIT_DIM)

## Saving the Seq2Seq models
Utilities.saveModel(seq2seq_model, 'seq2seq_model')
Utilities.saveModel(encoder_model, 'encoder_model')
Utilities.saveModel(decoder_model, 'decoder_model')
#
# Utilities.saveDict(max_len_dict, 'max_len_dict')
# Utilities.saveDict(input_tokenizer, 'input_tokenizer')
# Utilities.saveDict(translated_word2idx, 'translated_word2idx')
# Utilities.saveDict(translated_idx2word, 'translated_idx2word')

## Fitting the attention model
# at_encoder_embedding_layer, at_encoder_lstm_layer, at_decoder_embedding_layer, at_encoder_input, at_decoder_input, at_encoder_output, at_decoder_input_x = Models.usingAttention(embedding_matrix_input, max_len_input, max_len_translated, num_words_input, num_words_translated, EMBEDDING_DIM, UNIT_DIM)
# at_repeat_layer, at_concat_layer, at_dense1_layer, at_dense2_layer, at_softmax_layer, at_dot_layer = Models.defGlobalLayers(max_len_input)
# attention_model, at_decoder_lstm_layer, at_decoder_dense_layer, context_prev_word_concat_layer, decoder_initial_s, decoder_initial_c = Models.buildAttentionModel(num_words_translated, max_len_translated, at_encoder_input, at_decoder_input, at_encoder_output, at_decoder_input_x, UNIT_DIM, NUM_SAMPLES, input_seq, translated_input_seq, translated_output_onehot, BATCH_SIZE, EPOCHS, at_repeat_layer, at_concat_layer, at_dense1_layer, at_dense2_layer, at_softmax_layer, at_dot_layer)
# at_encoder_model, at_decoder_model = Models.attentionSamplingModel(at_encoder_input, at_encoder_output, max_len_input, decoder_initial_s, decoder_initial_c, at_decoder_embedding_layer, at_decoder_lstm_layer, at_decoder_dense_layer, context_prev_word_concat_layer, at_repeat_layer, at_concat_layer, at_dense1_layer, at_dense2_layer, at_softmax_layer, at_dot_layer, UNIT_DIM)
#
# ## Saving the Seq2Seq models
# Utilities.saveModel(attention_model, 'attention_model')
# Utilities.saveModel(at_encoder_model, 'at_encoder_model')
# Utilities.saveModel(at_decoder_model, 'at_decoder_model')
