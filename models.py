import numpy as np
import pandas as pd
import keras
from keras.models import Model
from keras.layers import Embedding, Input, Dense
from keras.layers import LSTM, GRU
from keras.optimizers import Adam

class Models:
    def usingSeq2Seq(embedding_matrix, max_len_input, max_len_translated, num_words_input, num_words_translated, embedding_dim, unit_dim, input_seq, translated_input_seq, translated_output_onehot, batch_size, epochs):

        encoder_embedding_layer = Embedding(embedding_matrix.shape[0],
                                    embedding_matrix.shape[1],
                                    weights = [embedding_matrix],
                                    trainable = False)

        encoder_input = Input(shape = (max_len_input,))
        x = encoder_embedding_layer(encoder_input)

        encoder_lstm_layer = LSTM(unit_dim, return_state = True, dropout = 0.4)
        encoder_output, h, c = encoder_lstm_layer(x)

        encoder_states = [h, c]

        decoder_input = Input(shape = (max_len_translated,))
        decoder_embedding_layer = Embedding(num_words_translated, embedding_dim)

        x2 = decoder_embedding_layer(decoder_input)

        decoder_lstm_layer = LSTM(unit_dim, return_sequences = True, return_state = True, dropout = 0.4)
        x2, _, _ = decoder_lstm_layer(x2, initial_state = encoder_states)

        decoder_dense_layer = Dense(num_words_translated, activation = 'softmax')
        decoder_output = decoder_dense_layer(x2)

        seq2seq_model = Model([encoder_input, decoder_input], decoder_output)

        seq2seq_model.compile(loss = 'categorical_crossentropy',
                        optimizer = Adam(lr = 1e-3, decay = 1e-6),
                        metrics = ['accuracy'])

        seq2seq_model.fit([input_seq, translated_input_seq],
                    translated_output_onehot,
                    batch_size = batch_size,
                    epochs = epochs,
                    validation_split = 0.1)

        return seq2seq_model, encoder_input, encoder_states, decoder_embedding_layer, decoder_lstm_layer, decoder_dense_layer

    def samplingModel(encoder_input, encoder_states, decoder_embedding_layer, decoder_lstm_layer, decoder_dense_layer, unit_dim):

        encoder_model = Model(encoder_input, encoder_states)

        decoder_model_input = Input(shape = (1,))
        decoder_model_input_h = Input(shape = (unit_dim,))
        decoder_model_input_c = Input(shape = (unit_dim,))
        decoder_model_input_states = [decoder_model_input_h, decoder_model_input_c]

        decoder_model_x = decoder_embedding_layer(decoder_model_input)
        decoder_model_x, decoder_model_output_h, decoder_model_output_c = decoder_lstm_layer(decoder_model_x, initial_state = decoder_model_input_states)
        decoder_model_output = decoder_dense_layer(decoder_model_x)
        decoder_model_output_states = [decoder_model_output_h, decoder_model_output_c]

        decoder_model = Model([decoder_model_input] + decoder_model_input_states,
                            [decoder_model_output] + decoder_model_output_states)

        return encoder_model, decoder_model

    def sampleFromSamplingModel(input_seq, encoder_model, decoder_model, translated_word2idx, translated_idx2word, max_len_translated):

        output_seq = []

        input_states = encoder_model.predict(input_seq)
        decoder_input = np.zeros((1, 1))
        decoder_input[0, 0] = translated_word2idx['<sos>']
        eos_idx = translated_word2idx['<eos>']

        for i in range(max_len_translated):
            output, h, c = decoder_model.predict([decoder_input] + input_states)

            idx = np.argmax(output[0, 0, :])

            if idx == eos_idx:
                break

            if idx > 0:
                output_seq.append(translated_idx2word[idx])

            decoder_input[0, 0] = idx
            input_states = [h, c]

        return ''.join(output_seq)
