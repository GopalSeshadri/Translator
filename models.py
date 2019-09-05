import numpy as np
import pandas as pd
import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Softmax
from keras.layers import Embedding, LSTM, GRU, Bidirectional
from keras.layers import RepeatVector, Concatenate, Dot, Lambda
from keras.optimizers import Adam, RMSprop
import keras.backend as K

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
                        optimizer = 'rmsprop',
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

    def softmaxOverT(x):
        e = K.exp(x - K.max(x, axis = 1, keepdims = True))
        s = K.sum(e, axis = 1, keepdims = True)
        softmax = e/s
        return softmax

    def usingAttention(embedding_matrix, max_len_input, max_len_translated, num_words_input, num_words_translated, embedding_dim, unit_dim):
        at_encoder_embedding_layer = Embedding(embedding_matrix.shape[0],
                                    embedding_matrix.shape[1],
                                    weights = [embedding_matrix],
                                    trainable = False)

        at_encoder_input = Input(shape = (max_len_input,))
        x1 = at_encoder_embedding_layer(at_encoder_input)

        at_encoder_lstm_layer = Bidirectional(LSTM(unit_dim, return_sequences = True, dropout = 0.5))
        at_encoder_output = at_encoder_lstm_layer(x1)


        # These decoder inputs are the teacher forcing inputs
        at_decoder_input = Input(shape = (max_len_translated,))
        at_decoder_embedding_layer = Embedding(num_words_translated, embedding_dim)
        at_decoder_input_x = at_decoder_embedding_layer(at_decoder_input)

        return at_encoder_embedding_layer, at_encoder_lstm_layer, at_decoder_embedding_layer, at_encoder_input, at_decoder_input, at_encoder_output, at_decoder_input_x

    def defGlobalLayers(max_len_input):
        # Defining global layers
        at_repeat_layer = RepeatVector(max_len_input) #To repeat previous decoder states Tx times
        at_concat_layer = Concatenate(axis = -1) # To concatenate repeated s-1 (Tx, UD2) and h (Tx, 2xUD1) along time axis
        at_dense1_layer = Dense(24, activation = 'tanh')
        at_dense2_layer = Dense(1)
        at_softmax_layer = Softmax(axis = 1)
        at_dot_layer = Dot(axes = 1)

        return at_repeat_layer, at_concat_layer, at_dense1_layer, at_dense2_layer, at_softmax_layer, at_dot_layer

    def oneStepOfAttention(h, prev_s, at_repeat_layer, at_concat_layer, at_dense1_layer, at_dense2_layer, at_softmax_layer, at_dot_layer):
        prev_s = at_repeat_layer(prev_s)
        x = at_concat_layer([h, prev_s])
        x = at_dense1_layer(x)
        x = at_dense2_layer(x)
        alphas = at_softmax_layer(x)
        context = at_dot_layer([alphas, h])
        return context

    def buildAttentionModel(num_words_translated, max_len_translated, at_encoder_input, at_decoder_input, at_encoder_output, at_decoder_input_x, unit_dim, num_samples, input_seq, translated_input_seq, translated_output_onehot, batch_size, epochs, at_repeat_layer, at_concat_layer, at_dense1_layer, at_dense2_layer, at_softmax_layer, at_dot_layer):
        at_decoder_lstm_layer = LSTM(unit_dim, return_state = True)
        at_decoder_dense_layer = Dense(num_words_translated, activation = 'softmax')
        decoder_initial_s = Input(shape = (unit_dim,))
        decoder_initial_c = Input(shape = (unit_dim,))
        context_prev_word_concat_layer = Concatenate(axis = 2)

        s = decoder_initial_s
        c = decoder_initial_c
        output_list = []
        for ty in range(max_len_translated):
            context = Models.oneStepOfAttention(at_encoder_output, s, at_repeat_layer, at_concat_layer, at_dense1_layer, at_dense2_layer, at_softmax_layer, at_dot_layer)
            lambda_layer = Lambda(lambda x: x[:, ty:ty+1])
            x_ty = lambda_layer(at_decoder_input_x)
            x = context_prev_word_concat_layer([context, x_ty])
            o, s, c = at_decoder_lstm_layer(x, initial_state = [s, c])
            decoder_output = at_decoder_dense_layer(o)
            output_list.append(decoder_output)

        stack_transpose_layer = Lambda(lambda x : K.permute_dimensions(K.stack(x), pattern = (1, 0, 2))) # We have to stack the output_list to convert it into a tensor
        outputs = stack_transpose_layer(output_list)

        attention_model = Model(inputs = [at_encoder_input, at_decoder_input, decoder_initial_s, decoder_initial_c], outputs = outputs)
        attention_model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        input_decoder_states = np.zeros((num_samples, unit_dim))
        attention_model.fit([input_seq, translated_input_seq, input_decoder_states, input_decoder_states],
                    translated_output_onehot,
                    batch_size = batch_size,
                    epochs = epochs,
                    validation_split = 0.1)
        return attention_model, at_decoder_lstm_layer, at_decoder_dense_layer, context_prev_word_concat_layer, decoder_initial_s, decoder_initial_c

    def attentionSamplingModel(at_encoder_input, at_encoder_output, max_len_input, decoder_initial_s, decoder_initial_c, at_decoder_embedding_layer, at_decoder_lstm_layer, at_decoder_dense_layer, context_prev_word_concat_layer, at_repeat_layer, at_concat_layer, at_dense1_layer, at_dense2_layer, at_softmax_layer, at_dot_layer, unit_dim):
        encoder_model = Model(at_encoder_input, at_encoder_output)

        # Defining T=1 decoder sampling model
        encoder_output_as_decoder_input = Input(shape = (max_len_input, unit_dim * 2, ))
        decoder_input_single = Input(shape = (1,))
        decoder_input_single_x = at_decoder_embedding_layer(decoder_input_single)

        # Getting context for the given h and s
        context = Models.oneStepOfAttention(encoder_output_as_decoder_input, decoder_initial_s, at_repeat_layer, at_concat_layer, at_dense1_layer, at_dense2_layer, at_softmax_layer, at_dot_layer)
        x = context_prev_word_concat_layer([context, decoder_input_single_x])
        x, s, c = at_decoder_lstm_layer(x, initial_state = [decoder_initial_s, decoder_initial_c])
        output = at_decoder_dense_layer(x)

        decoder_model = Model(inputs = [decoder_input_single, encoder_output_as_decoder_input, decoder_initial_c, decoder_initial_s], outputs = [output, s, c])

        return encoder_model, decoder_model

    def sampleFromAttentionSamplingModel(input_seq, at_encoder_model, at_decoder_model, translated_word2idx, translated_idx2word, max_len_translated, unit_dim):

        output_seq = []

        encoder_output = at_encoder_model.predict(input_seq)
        decoder_input = np.zeros((1, 1))
        decoder_input[0, 0] = translated_word2idx['<sos>']
        eos_idx = translated_word2idx['<eos>']

        s = np.zeros((1, unit_dim))
        c = np.zeros((1, unit_dim))

        for i in range(max_len_translated):
            output, s, c = at_decoder_model.predict([decoder_input, encoder_output, s, c])

            idx = np.argmax(output.flatten())

            if idx == eos_idx:
                break

            if idx > 0:
                output_seq.append(translated_idx2word[idx])

            decoder_input[0, 0] = idx

        return ''.join(output_seq)
