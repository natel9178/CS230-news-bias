"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
from numpy import asarray
import numpy as np
import tensorflow.contrib.eager as tfe

EMBEDDING_DIM = 100

def load_words(path_words):
    return list(line.strip() for line in open(path_words))

def load_glove_embedding(path_txt):
    embeddings_index = dict()
    f = open(path_txt) # f = open('../data/GloVe/glove.6B.100d.txt')
    # words = load_words(path_words)
    for line in f:
        values = line.split()
        word = values[0]
        # if word in words:
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors' % len(embeddings_index))
    return embeddings_index

def generate_embedding_matrix(words_list, embeddings_index):
    embedding_matrix = np.zeros((len(words_list) + 1, EMBEDDING_DIM))
    for i, word in enumerate(words_list):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_dataset_from_text(path_txt, vocab):
    """Create tf.data Instance from txt file

    Args:
        path_txt: (string) path containing one example per line
        vocab: (tf.lookuptable)

    Returns:
        dataset: (tf.Dataset) yielding list of ids of tokens for each example
    """
    
    with tf.device('/cpu:0'):
        # Load txt file, one example per line
        dataset = tf.data.TextLineDataset(path_txt)

        # Convert line into list of tokens, splitting by white space
        dataset = dataset.map(lambda string: tf.string_split([string]).values)

        # Lookup tokens to return their ids
        dataset = dataset.map(lambda tokens: (vocab.lookup(tokens), tf.size(tokens)))

    return dataset

def input_fn(mode, articles, labels, params):
    """Input function for NER

    Args:
        mode: (string) 'train', 'eval' or any other mode you can think of
                     At training, we shuffle the data and have multiple epochs
        articles: (tf.Dataset) yielding list of ids of words
        datasets: (tf.Dataset) yielding list of ids of tags
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    # Load all the dataset in memory for shuffling is training
    is_training = (mode == 'train')
    buffer_size = params.buffer_size if is_training else 1
    with tf.device('/cpu:0'):
        # Zip the sentence and the labels together
        dataset = tf.data.Dataset.zip((articles, labels))

        # Create batches and pad the articles of different length
        padded_shapes = (([10000], []), [])


        padding_values = (params.id_pad_word,   # sentence padded on the right with id_pad_word
                            None)

        
        dataset = (dataset
            .shuffle(buffer_size=buffer_size)
            .padded_batch(4, padded_shapes=padded_shapes) #, padding_values=padding_values) #.batch(30)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    ((article, article_length), labels) = iterator.get_next() # ((sentence, sentence_lengths), (labels, _))
    init_op = iterator.initializer

    # # DEBUG TODO: Remove
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.tables_initializer(name='init_all_tables'))
    #     sess.run(init_op)
    #     for i in range(5):
    #         a, b = sess.run(next_element)
    #         print(a)

    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'article': article,
        'labels': labels,
        'article_length': article_length,
        'iterator_init_op': init_op
    }

    return inputs
