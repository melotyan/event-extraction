from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Dense, Input, Bidirectional, LSTM, Conv1D, Flatten, MaxPool1D
from keras.layers import GlobalMaxPool1D, Concatenate, Dropout, Reshape
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import RandomUniform
from attention import Attention
import tensorflow as tf
import numpy as np
from metrics import auc, softmax_evaluation, sigmoid_evaluation
from init_training_data import get_multi_conv_data
import argparse
from sklearn import preprocessing
from tensorflow.python.framework import dtypes
from keras.utils.np_utils import to_categorical

from keras import backend as K


vocabulary_size = 20000
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
MAX_SEQUENCE_LENGTH = 50
CONTEXT_SIZE = 3
BATCH_SIZE = 32
WORD_TO_VECTOR_PATH = 'en.vec'
MODEL_SAVE_PATH = 'multi-convolution.h5'
num_lstm = 100
KERNEL_SIZE = 3
FILTER_NUM = 32
model_dir = 'multi_tf'



def get_data():
    #数据获取
    words, train_x, train_lexical, train_pos, train_y = get_multi_conv_data('multi-conv.json')
    words = words[0:200000]
    train_x = train_x[0:200000]
    train_lexical = train_lexical[0:200000]
    train_pos = train_pos[0:200000]
    train_y = train_y[0:200000]

    #转成category格式
    # train_y = tf.one_hot(train_y, 2, 1, 0)
    # print (train_y[0:30])
    # print ('总样本数', sum(np.argmax(a) for a in train_y))
    # 划分一部分做盲测集
    test_x, test_local, test_pos, test_y = train_x[-20000:], train_lexical[-20000:], train_pos[-20000:], train_y[-20000:]
    train_x, train_lexical, train_pos, train_y = train_x[:-20000], train_lexical[:-20000], train_pos[:-20000], train_y[:-20000]
    print ('train shape', np.asarray(train_x).shape)#, train_pos.shape, train_y.shape)

    #词-数
    tokenizer = text.Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(words)
    word_index = tokenizer.word_index

    #生成数字序列
    train_sequence = tokenizer.texts_to_sequences(train_x)
    train_lexical_sequence = tokenizer.texts_to_sequences(train_lexical)
    train_pos = tokenizer.texts_to_sequences(train_pos)
    test_sequence = tokenizer.texts_to_sequences(test_x)
    test_local_sequence = tokenizer.texts_to_sequences(test_local)
    test_pos = tokenizer.texts_to_sequences(test_pos)

    #补齐
    train_data = pad_sequences(sequences=train_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    train_lexical_feature = pad_sequences(sequences=train_lexical_sequence, maxlen=CONTEXT_SIZE, padding='post')
    train_pos = pad_sequences(sequences=train_pos, maxlen=MAX_SEQUENCE_LENGTH)
    test_data = pad_sequences(sequences=test_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    test_local_feature = pad_sequences(sequences = test_local_sequence, maxlen=CONTEXT_SIZE, padding='post')
    test_pos = pad_sequences(sequences=test_pos, maxlen=MAX_SEQUENCE_LENGTH)

    print ('train_data shape', train_data.shape, 'test data shape', test_data.shape)

    return train_data, train_lexical_feature, train_pos, train_y, test_data, test_local_feature, test_pos, test_y, tokenizer, test_x

def load_embedding_matrix(word_index):
    #读取预训练的词向量
    embeddings_index = {}
    with open(WORD_TO_VECTOR_PATH) as f:
        #第一行不是向量
        f.readline()
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vec
    print('Total %s word vectors.' % len(embeddings_index))

    #提取需要的词向量矩阵
    nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
    embeddings_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i - 1] = embedding_vector
    print('embedding shape', embeddings_matrix.shape)
    return nb_words, embeddings_matrix

def build_netword(features, nb_words, embeddings_matrix):
    '''
    build the multi-pooling convolution network
    :param nb_words:
    :param embeddings_matrix:
    :return:
    '''
    #定义网络层
    sentence_input = features['sentence']
    lexical_input = features['lexical']
    position_input = features['pos']
    # sentence_input = tf.feature_column.input_layer(features=features, feature_columns=[tf.feature_column.numeric_column('sentence', dtype=dtypes.int32)])
    # lexical_input = tf.feature_column.input_layer(features= features, feature_columns=[tf.feature_column.numeric_column('lexical', dtype=dtypes.int32)])
    # position_input = tf.feature_column.input_layer(features= features, feature_columns=[tf.feature_column.numeric_column('pos', dtype=dtypes.int32)])
    # sentence_input = tf.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='sentence_input')
    # lexical_input = tf.layers.Input(shape=(CONTEXT_SIZE,), dtype='int32', name='lexical_input')
    # position_input = tf.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='position_input')
    print ('embedding matrix', type(embeddings_matrix))# embeddings_matrix.shape, embeddings_matrix[0:1])
    word_embedding = tf.get_variable(initializer=embeddings_matrix, name='word-embedding', dtype=tf.float64)
    random_embedding = tf.Variable(tf.random_uniform([nb_words, 10], -1.0, 1.0, dtype=tf.float64))
    sentence_embedding = tf.nn.embedding_lookup(word_embedding, sentence_input)
    position_embedding = tf.nn.embedding_lookup(word_embedding, position_input)
    print ('embedding shape', sentence_embedding.shape, position_embedding.shape)
    # cnn_layer1 = tf.layers.Conv1D()

    x1 = tf.concat([sentence_embedding, position_embedding, position_embedding], axis=-1)
    print ('x1 shape', x1.shape, x1.dtype)
    x1 = tf.cast(x1, tf.float32)
    x1 = tf.layers.conv1d(x1, filters=FILTER_NUM, kernel_size=(KERNEL_SIZE))
    # x1 = cnn_layer1(x1)
    print ('x1 shape after cnn', x1.shape)
    tf.layers.MaxPooling1D
    x1 = tf.layers.max_pooling1d(x1, pool_size=MAX_SEQUENCE_LENGTH - KERNEL_SIZE + 1, strides=1)
    print ('x1 shape after maxpooling', x1.shape)
    x1 = tf.reshape(x1, [-1, FILTER_NUM])
    x2 = tf.nn.embedding_lookup(word_embedding, lexical_input)
    x2 = tf.reshape(x2, (-1, CONTEXT_SIZE * EMBEDDING_DIM))
    x2 = tf.cast(x2, tf.float32)
    print ('x2 shape dtype', x2.shape, x2.dtype)
    x = tf.concat([x1, x2], axis=-1)
    print ('x shape', x.shape)
    x = tf.layers.dense(x, 100, activation=tf.nn.relu)
    x = tf.layers.dense(x, 100, activation=tf.nn.relu)
    x = tf.layers.dense(x, 100, activation=tf.nn.relu)
    x = tf.layers.dense(x, 100, activation=tf.nn.relu)
    x = tf.layers.dropout(x, 0.5)
    x = tf.layers.dense(x, 2, activation=tf.nn.softmax)

    return x

def model_fn(features, labels, mode, params):
    # labels = tf.one_hot(labels, 2)
    logits = build_netword(features, params['nb_words'], params['embedding_matrix'])
    pred_classes = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions = pred_classes)

    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=tf.one_hot(labels, 2), pos_weight=8))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    # auc_op = auc(y_pred=pred_classes, y_true=labels)
    print ('label shape', labels.shape)

    precision = tf.metrics.precision(labels=labels, predictions=pred_classes, name='precision')
    recall = tf.metrics.recall(labels=labels, predictions=pred_classes, name='recall')
    accuracy = tf.metrics.accuracy(labels=labels, predictions=pred_classes, name='accuracy')
    # if (recall > 0):
    #     f_score = 2 * precision * recall / (recall + precision)
    estim_specs = tf.estimator.EstimatorSpec(
        mode = mode,
        predictions = pred_classes,
        train_op = train_op,
        eval_metric_ops = {'accuracy':accuracy, 'precision':precision, 'recall':recall},
        loss=loss,

    )
    return estim_specs

def input_fn(features, label, is_train = True):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), label))
    print ('dataset shape', dataset.output_shapes)
    return dataset#.make_one_shot_iterator().get_next()


def main():
    # Define the input function for training
    train_data, train_lexical_feature, train_pos, train_y, test_data, test_lexical_feature, test_pos, test_y, tokenizer, test_x = get_data()
    nb_worbs, embedding_matrix = load_embedding_matrix(tokenizer.word_index)
    #train input
    feature = {}
    print ('embedding matrix type', type(embedding_matrix))
    feature['sentence'] = train_data
    feature['lexical'] = train_lexical_feature
    feature['pos'] = train_pos
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=feature, y=np.asarray(train_y), batch_size=BATCH_SIZE, num_epochs=10, shuffle=True)
    #evaluate input
    test_feature = {}
    test_feature['sentence'] = test_data
    test_feature['lexical'] = test_lexical_feature
    test_feature['pos'] = test_pos
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=test_feature, y=np.asarray(test_y), batch_size=BATCH_SIZE,
                                                       num_epochs=1, shuffle=False)

    model = tf.estimator.Estimator(model_fn=model_fn, model_dir='multi_tf',
                                   params={'embedding_matrix': embedding_matrix, 'nb_words': nb_worbs})
    for i in range(10):
        model.train(input_fn = train_input_fn)

        e = model.evaluate(eval_input_fn)
        print("precision", e['precision'], 'recall', e['recall'])

        # res = model.predict(eval_input_fn)
        # count = 0
        # for r in res:
        #     count += 1
        #     print (type(r), r)
        # print ('count', count)
        # print (r['probabilities'][1] for r in res)



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

