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

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.metrics import precision_recall_fscore_support

#import matplotlib.pyplot as plt

if __name__ == '__main__':
    # print('Indexing word vectors.')
    # embeddings_index = index_glove_embeddings(GLOVE_DIR)

    # print('Processing text dataset')
    x_train = load_text_dataset(os.path.join(TEXT_DATA_DIR,'train/articles.txt'))
    x_dev = load_text_dataset(os.path.join(TEXT_DATA_DIR,'dev/articles.txt'))
    y_dev = load_text_dataset(os.path.join(TEXT_DATA_DIR,'dev/tags.txt'))
    # x_test = load_text_dataset(os.path.join(TEXT_DATA_DIR,'test/articles.txt'))
    # y_test = load_text_dataset(os.path.join(TEXT_DATA_DIR,'test/tags.txt'))
    print('Found %s texts.' % len(x_dev))

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(x_train + x_dev)
    x_dev = tokenizer.texts_to_sequences(x_dev)
    x_dev = pad_sequences(x_dev, maxlen=MAX_SEQUENCE_LENGTH)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    y_dev = np.asarray(y_dev)

    y_dev = y_dev.astype(np.int)

    #embedding_layer = create_embedding_layer(word_index, embeddings_index)
    
    # if(MODEL == 'lstm'):
    #     MODEL_FINAL_DIR = LSTM_FINAL_DIR
    # elif(MODEL == 'conv'):
    #     MODEL_FINAL_DIR = CONV_FINAL_DIR

    fig_num = 1

    for suite in TESTS:
        model_name, num_layers = suite
        MODEL_FINAL_DIR = '{}{}{}{}{}'.format('experiments/weights/',NUM_CLASSES,num_layers,model_name,'_weights.final.hdf5')

        if os.path.exists(MODEL_FINAL_DIR):
            print('Loading previous model weights.')
            #model.load_weights(MODEL_CP_DIR)
            model = load_model(MODEL_FINAL_DIR)
        else:
            print('Model is blank')
            exit

        y_preds = model.predict(x_dev, verbose=1)

        fpr, tpr, thresholds = roc_curve(y_dev, y_preds)
        roc_auc = auc(fpr, tpr)
        np.savetxt('{}{}fpr.out'.format(num_layers,model_name), fpr)
        np.savetxt('{}{}tpr.out'.format(num_layers,model_name), tpr)
        print('{}{} auc: {}'.format(num_layers,model_name,roc_auc))

        y_preds = (y_preds > 0.5)

        score = precision_recall_fscore_support(y_dev, y_preds)
        print('{}{} precision recall fscore support: {}'.format(num_layers,model_name,score))

    #     # Plot ROC curve
    #     plt.figure(fig_num)
    #     fig_num = fig_num + 1
    #     plt.plot(fpr, tpr, label=('ROC curve {}{}{}{}{} (area = %0.3f)' % roc_auc).format('for ',model_name,' with ',num_layers, ' layers'))
    #     plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.0])
    #     plt.xlabel('False Positive Rate or (1 - Specifity)')
    #     plt.ylabel('True Positive Rate or (Sensitivity)')
    #     plt.title('Receiver Operating Characteristic')
    #     plt.legend(loc="lower right")
    # plt.show()