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
import pickle

MODEL_FINAL_DIR = './experiments/weights/3_class_4conv_weights.final.hdf5'

if __name__ == '__main__':
    print('Indexing word vectors.')
    embeddings_index = []
    with open('embeddings_index.pickle', 'rb') as handle:
        embeddings_index = pickle.load(handle)

    # embeddings_index = index_glove_embeddings(GLOVE_DIR)

    # with open('embeddings_index.pickle', 'wb') as handle:
    #     pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Processing text dataset')
    # x_train = load_text_dataset(os.path.join(TEXT_DATA_DIR,'train/articles.txt'))
    # x_dev = load_text_dataset(os.path.join(TEXT_DATA_DIR,'dev/articles.txt'))
    x_test = load_text_dataset(os.path.join(TEXT_DATA_DIR,'example/articles.txt'))
    y_test = load_text_dataset(os.path.join(TEXT_DATA_DIR,'example/tags.txt'))
    print('Found %s texts.' % len(x_test))

    # tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    # tokenizer.fit_on_texts(x_train + x_dev)

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # with open('tokenizer.pickle', 'wb') as handle:
    #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    y_test = to_categorical(np.asarray(y_test))

    embedding_layer = create_embedding_layer(word_index, embeddings_index)
    
    if os.path.exists(MODEL_FINAL_DIR):
        print('Loading previous model weights.')
        model = load_model(MODEL_FINAL_DIR)
    else:
        print('Model is blank')
        exit

    scores = model.evaluate(x_test, y_test, verbose=1)
    print("Acheived result on example set - %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    np.set_printoptions(suppress=True)
    predictions = model.predict(x_test) * 100
    argmax = np.argmax(predictions, axis = 1)

    print(predictions)
    print("Argmax of Preds: \t {}".format(argmax))
    print("Text Labels:\t\t {}".format(np.argmax(y_test, axis = 1)))
    print('\n')
    