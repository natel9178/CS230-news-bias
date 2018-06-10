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

from train import *

from keras.models import load_model

# BASE_DIR = ''
# GLOVE_DIR = './data/GloVe/glove.6B.100d.txt'
# TEXT_DATA_DIR = './data/kaggle'
# TENSORBOARD_BASE_DIR = 'experiments/tensorboard'
# #MODEL_CP_DIR = 'experiments/weights/weights.best.hdf5'
# MAX_SEQUENCE_LENGTH = 1000
# MAX_NUM_WORDS = 20000
# EMBEDDING_DIM = 100
# LSTM_FINAL_DIR = 'experiments/weights/lstm_weights.final.hdf5'
# CONV_FINAL_DIR = 'experiments/weights/conv_weights.final.hdf5'
# #MODEL = 'lstm'


# def index_glove_embeddings(fname):
#     # first, build index mapping words in the embeddings set
#     # to their embedding vector

#     embeddings_index = {}
#     with open(fname) as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             coefs = np.asarray(values[1:], dtype='float32')
#             embeddings_index[word] = coefs

#     print('Found %s word vectors.' % len(embeddings_index))
#     return embeddings_index

# def load_text_dataset(fname, max_index = 9999999):
#     datas = []  # list of text samples
#     i = 0
#     with open(fname) as f:
#         for line in f:
#             datas.append(line)
#             if i >= max_index:
#                 break
#             i += 1
#     return datas

# def create_embedding_layer(word_index):
#     print('Preparing embedding matrix.')

#     # prepare embedding matrix
#     num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
#     embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
#     for word, i in word_index.items():
#         if i >= MAX_NUM_WORDS:
#             continue
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             # words not found in embedding index will be all-zeros.
#             embedding_matrix[i] = embedding_vector

#     # load pre-trained word embeddings into an Embedding layer
#     # note that we set trainable = False so as to keep the embeddings fixed
#     return Embedding(num_words, EMBEDDING_DIM,
#                                 weights=[embedding_matrix],
#                                 input_length=MAX_SEQUENCE_LENGTH,
#                                 trainable=False)


if __name__ == '__main__':
    print('Indexing word vectors.')
    embeddings_index = index_glove_embeddings(GLOVE_DIR)

    print('Processing text dataset')
    x_train = load_text_dataset(os.path.join(TEXT_DATA_DIR,'train/articles.txt'))
    x_dev = load_text_dataset(os.path.join(TEXT_DATA_DIR,'dev/articles.txt'))
    x_test = load_text_dataset(os.path.join(TEXT_DATA_DIR,'test/articles.txt'))
    y_test = load_text_dataset(os.path.join(TEXT_DATA_DIR,'test/tags.txt'))
    print('Found %s texts.' % len(x_test))

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(x_train + x_dev)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    y_test = to_categorical(np.asarray(y_test))

    embedding_layer = create_embedding_layer(word_index, embeddings_index)
    
    # if(MODEL == 'lstm'):
    #     MODEL_FINAL_DIR = LSTM_FINAL_DIR
    # elif(MODEL == 'conv'):
    #     MODEL_FINAL_DIR = CONV_FINAL_DIR
    if os.path.exists(MODEL_FINAL_DIR):
        print('Loading previous model weights.')
        #model.load_weights(MODEL_CP_DIR)
        model = load_model(MODEL_FINAL_DIR)
    else:
        print('Model is blank')
        exit

    scores = model.evaluate(x_test, y_test, verbose=1)
    
    print("Acheived result on test set - %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    