"""Train the model"""

from __future__ import print_function

import os
import sys
import numpy as np
import time
from metrics import mcor,recall,f1,precision

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, Dropout, Activation
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras.backend as K

BASE_DIR = ''
GLOVE_DIR = './data/GloVe/glove.6B.100d.txt'
TEXT_DATA_DIR = './data/kaggle'
TENSORBOARD_BASE_DIR = 'experiments/tensorboard'
MODEL_CP_DIR = 'experiments/weights/weights.best.hdf5'
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

def index_glove_embeddings(fname):
    # first, build index mapping words in the embeddings set
    # to their embedding vector

    embeddings_index = {}
    with open(fname) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def load_text_dataset(fname, max_index = 9999999):
    datas = []  # list of text samples
    i = 0
    with open(fname) as f:
        for line in f:
            datas.append(line)
            if i >= max_index:
                break
            i += 1
    return datas

def create_embedding_layer(word_index):
    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    return Embedding(num_words, EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

def model_fn(model_type, embedding_layer):
    print('Creating model.')
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    if model_type == 'lstm':
        X = LSTM(128, return_sequences=True)(embedded_sequences)
        X = Dropout(0.2)(X)
        X = LSTM(128, return_sequences=False)(X)
        X = Dropout(0.2)(X)
        X = Dense(1)(X)
        preds = Activation('sigmoid')(X)
    else:
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(1, activation='sigmoid')(x)

    return Model(sequence_input, preds)

if __name__ == '__main__':
    print('Indexing word vectors.')
    embeddings_index = index_glove_embeddings(GLOVE_DIR)

    print('Processing text dataset')
    x_test = load_text_dataset(os.path.join(TEXT_DATA_DIR,'test/articles.txt'))
    y_test = load_text_dataset(os.path.join(TEXT_DATA_DIR,'test/tags.txt'))
    print('Found %s texts.' % len(x_test))

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(x_test)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    y_test = np.asarray(y_test)

    embedding_layer = create_embedding_layer(word_index)
    model = model_fn('lstm', embedding_layer)

    if os.path.exists(MODEL_CP_DIR):
        print('Loading previous model weights.')
        model.load_weights(MODEL_CP_DIR)
    else:
        print('Model is blank')
        exit

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['acc'])
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("Acheived result on test set - %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    