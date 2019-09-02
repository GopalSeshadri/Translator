import numpy as np
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go
import dash
import dash_core_components as core
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import sys
from preprocess import Preprocess
import pickle
from keras.models import model_from_json
from keras import backend as K
from utilities import Utilities
from models import Models
from keras import backend as K

app = dash.Dash()

app.layout = html.Div([
    html.Div([
        html.H1('English to Japanese Translation', style = {'fontFamily' : 'Helvetica',
                                                        'textAlign' : 'center',
                                                        'width' : '100%'})
    ], style = {'display' : 'flex'}),

    html.Div([
        html.Div([
            html.H3('Enter Text',
                style = {'fontFamily' : 'Helvetica',
                        'fontSize' : 16,
                        'display' : 'inline-block',
                        'float' : 'left'}),

            core.Textarea(id = 'input-text',
                placeholder = 'Enter something here ...',
                style = {'width': '500px',
                        'height' : '200px',
                        'display' : 'inline-block',
                        'float' : 'left'}
            )
        ], style = {'width' : '40%',
                    'display' : 'inline-block',
                    'float' : 'left'}),

        html.Div([
            html.Button(id = 'submit-button', children = 'Translate',
                        style = {'width' : '80px',
                                'height' : '30px'})
        ], style = {'width' : '100%',
                    'paddingLeft' : '50px'})

    ], style = {'width' : '60%',
                'height' : '80%',
                'display' : 'flex',
                'float' : 'left',
                'paddingLeft' : '50px'}),

    html.Div([
        html.H3('Translated Text',
            style = {'fontSize' : 16, 'fontFamily' : 'Helvetica',
                    'display' : 'inline-block',
                    'float' : 'left'}),

        core.Textarea(id = 'translated-text',
                    placeholder = 'Translated text ...',
                    style = {'width': '500px',
                            'height' : '200px',
                            'display' : 'inline-block',
                            'float' : 'left'})

    ], style = {'width' : '20%',
                'display' : 'inline-block',
                'float' : 'left'})

], style = {'width' : '100%', 'height' : '100%'})


@app.callback(Output('translated-text', 'value'),
            [Input('submit-button', 'n_clicks')],
            [State('input-text', 'value')])
def affectTranslate(n_clicks, text):

        translated_output = ''
        if text != None and text != '':
            seq2seq_model = Utilities.loadModel('seq2seq_model')
            encoder_model = Utilities.loadModel('encoder_model')
            decoder_model = Utilities.loadModel('decoder_model')

            max_len_dict = Utilities.loadDict('max_len_dict')
            input_tokenizer = Utilities.loadDict('input_tokenizer')
            translated_word2idx = Utilities.loadDict('translated_word2idx')
            translated_idx2word = Utilities.loadDict('translated_idx2word')

            print(text, file=sys.stderr)
            text_list = []
            text_list.append(text)
            print(text_list, file=sys.stderr)

            text_seq = input_tokenizer.texts_to_sequences(text_list)
            text_seq = Preprocess.padInputSequences(text_seq, max_len_dict['max_len_input'])
            print(text_seq, file=sys.stderr)
            translated_output = Models.sampleFromSamplingModel(text_seq, encoder_model, decoder_model, translated_word2idx, translated_idx2word, max_len_dict['max_len_translated'])
            print(translated_output, file=sys.stderr)

            K.clear_session()

        return translated_output

if __name__ == '__main__':
    app.run_server()