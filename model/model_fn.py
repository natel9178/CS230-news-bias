"""Define the model."""

import tensorflow as tf
import keras 
import numpy as np
from keras import backend as K
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import GlobalMaxPooling1D

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 1000

def build_model(mode, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    article = inputs['article']
    embedding_matrix = inputs['glove']
    words_list = inputs['words_list']

    if params.model_version == 'lstm':
        # Get word embeddings for each token in the sentence
        model = Sequential()
        embedding_layer = Embedding(len(words_list) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH) # Model does not yet use glove embeddings for preliminary results
        model.add(embedding_layer)
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(128, 5, activation='relu'))
        # model.add(MaxPooling1D(35))
        # model.add(Flatten())
        model.add(GlobalMaxPooling1D())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation=None))
        
        
        # Compute logits from the output of the LSTM
        logits = model(article)
    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    return logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    article_length = inputs['article_length']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(mode, inputs, params)
        predictions = tf.cast(tf.greater(tf.sigmoid(logits),0.5), tf.float32)
        # predictions = tf.Print(predictions, [predictions], "predictions")
        # predictions = tf.Print(predictions, [tf.sigmoid(logits)], "logits")

    string_tensor = tf.string_to_number(labels)
    y_labels = tf.reshape(string_tensor, [-1, 1])
    # Define loss and accuracy (we need to apply a mask to account for padding)
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_labels)
    # mask = tf.sequence_mask(article_length)
    # losses = tf.boolean_mask(losses, mask)
    loss = tf.reduce_mean(losses)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=tf.string_to_number(labels), predictions=predictions),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    # with tf.Session() as sess:
    #     K.set_session(sess)
    #     # Initialize model variables
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(model_spec['variable_init_op'])
    #     sess.run(model_spec['iterator_init_op'])
    #     a = sess.run(string_tensor)
    #     b = sess.run(y_labels)
    #     c = sess.run(logits)

    #     d = 1

    return model_spec
