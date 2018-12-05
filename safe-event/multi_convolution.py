# encoding=utf8

from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Dense, Input, Bidirectional, LSTM, Conv1D, Flatten, MaxPool1D
from keras.layers import GlobalMaxPool1D, Concatenate, Dropout, Reshape
from keras.layers.core import RepeatVector
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import RandomUniform
import numpy as np
from metrics import auc, softmax_evaluation, sigmoid_evaluation
from init_training_data import get_multi_conv_data
from keras import backend as K
import tensorflow as tf


vocabulary_size = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
MAX_SEQUENCE_LENGTH = 50
CONTEXT_SIZE = 3
BATCH_SIZE = 32
WORD_TO_VECTOR_PATH = '../zh-word-vec100.txt'
MODEL_SAVE_PATH = 'multi-convolution3.h5'
num_lstm = 100
TEST_NUM = 10000


def save_word_embedding():
    # 数据获取
    words, train_x, train_lexical, train_pos, train_y = get_multi_conv_data('word-multi-conv-big.txt')
    words_test, test_x, test_lexical, test_pos, test_y = get_multi_conv_data('word-multi-conv-val.txt')

    word_sets = set(words + words_test)
    tokenizer = text.Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(word_sets)
    word_index = tokenizer.word_index
    with open('tokenizer-words.txt', 'w') as f:
        for word in word_sets:
            f.write(word + '\n')

    # 读取预训练的词向量
    embeddings_index = {}
    with open(WORD_TO_VECTOR_PATH, encoding='utf-8') as f:
        # 第一行不是向量
        f.readline()

        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vec
    print('Total %s word vectors.' % len(embeddings_index))

    # 提取需要的词向量矩阵
    nb_words = len(word_index) + 1
    embeddings_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= nb_words - 1:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i - 1] = embedding_vector
    #unknow的词向量
    embeddings_matrix[len(word_index)] = np.random.normal(size=EMBEDDING_DIM, loc=0, scale=0.05)
    print('embedding shape', embeddings_matrix.shape)
    vec = np.array(embeddings_matrix, dtype=np.float32)

    np.save('vec.npy', vec)

def prepare_train_data(tokenizer):
    #数据获取
    words, train_x, train_lexical, train_pos, train_y = get_multi_conv_data('word-multi-conv-big.txt')
    train_pos = np.asarray(train_pos)
    #转成category格式
    from keras.utils.np_utils import to_categorical
    train_y = to_categorical(train_y, num_classes=2)

    print ('总样本数', sum(np.argmax(a) for a in train_y))
    print ('train shape', np.asarray(train_x).shape, train_pos.shape, train_y.shape)
    print ('训练样本数', sum(np.argmax(a) for a in train_y))

    #生成数字序列
    train_sequence = tokenizer.texts_to_sequences(train_x)
    train_lexical_sequence = tokenizer.texts_to_sequences(train_lexical)
    #补齐
    train_data = pad_sequences(sequences=train_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    train_lexical_feature = pad_sequences(sequences=train_lexical_sequence, maxlen=CONTEXT_SIZE, padding='post')
    train_pos = pad_sequences(sequences=train_pos, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print ('before train pos shape', train_pos.shape)
    train_pos = train_pos[:,:,np.newaxis]
    print ('after train pos shape', train_pos.shape)
    print ('train_data shape', train_data.shape)

    return train_data, train_lexical_feature, train_pos, train_y

def prepare_validate_data(tokenizer, path='word-multi-conv-val.txt'):
    test_word, test_x, test_lexical, test_pos, test_y = get_multi_conv_data(path)
    test_pos = np.asarray(test_pos)

    from keras.utils.np_utils import to_categorical
    test_y = to_categorical(test_y, num_classes=2)

    # 生成数字序列
    test_sequence = tokenizer.texts_to_sequences(test_x)
    test_local_sequence = tokenizer.texts_to_sequences(test_lexical)

    # 补齐
    test_data = pad_sequences(sequences=test_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    test_local_feature = pad_sequences(sequences=test_local_sequence, maxlen=CONTEXT_SIZE, padding='post')
    test_pos = pad_sequences(sequences=test_pos, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    test_pos = test_pos[:,:,np.newaxis]
    print('test data shape', test_data.shape)
    print('tests label shape', np.asarray(test_y).shape, 'count', sum(np.argmax(a) for a in test_y))
    return test_data, test_local_feature, test_pos, test_y, test_x

def build_network():
    #读取词表,获取tokenizer
    words = []
    with open('tokenizer-words.txt') as f:
        for line in f:
            words.append(line.strip())
    tokenizer = text.Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(words)
    word_index = tokenizer.word_index
    # 读取预训练的词向量
    embeddings_index = {}
    with open(WORD_TO_VECTOR_PATH, encoding='utf-8') as f:
        # 第一行不是向量
        f.readline()

        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vec
    print('Total %s word vectors.' % len(embeddings_index))

    # 提取需要的词向量矩阵
    nb_words = len(word_index) + 1
    embeddings_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
    # unknow的词向量
    embeddings_matrix[0] = np.random.normal(size=EMBEDDING_DIM, loc=0, scale=0.05)

    #定义网络层
    sentence_embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embeddings_matrix],
                                input_length=MAX_SEQUENCE_LENGTH, trainable=False, mask_zero=False)
    lexical_embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embeddings_matrix],
                                input_length=CONTEXT_SIZE, trainable=False, mask_zero=False)
    # position_embedding_layer = Embedding(nb_words, 10, input_length=MAX_SEQUENCE_LENGTH, trainable=True, mask_zero=False)
    # position_embedding_layer = Dense(10, )

    cnn_layer1 = Conv1D(filters=32, kernel_size=(3))#, kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01))
    #构建模型
    sentence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='sentence_input')
    lexical_input = Input(shape=(CONTEXT_SIZE,), dtype='int32', name='lexical_input')
    position_input = Input(shape=(MAX_SEQUENCE_LENGTH,1), dtype='float32', name='position_input')

    sentence_embedding_sequence = sentence_embedding_layer(sentence_input)
    x2 = lexical_embedding_layer(lexical_input)

    # position_embedding_sequence = position_embedding_layer(position_input)
    # print ('position embedding shape', position_embedding_sequence.shape)
    print ('position input shape', position_input.shape)
    # position_input = tf.expand_dims(position_input, -1)
    x1 = Concatenate(axis=-1)([sentence_embedding_sequence, position_input, position_input])

    print ('x1 shape', x1.shape)
    x1 = cnn_layer1(x1)
    x1 = GlobalMaxPool1D()(x1)
    x2 = Reshape((3 * EMBEDDING_DIM,))(x2)
    print ('x2 x1 shape', x2.shape, x1.shape)
    x = Concatenate(axis=-1)([x1, x2])
    print('x shape', x.shape)
    for i in range(16):
        x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax', name='output')(x)
    print ('output x shape', x.shape)

    model = Model(inputs=[sentence_input, lexical_input, position_input], outputs=x)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=[auc])
    model.summary()

    return model, tokenizer

def train(model, train_data, train_lexical_feature, train_pos, train_y):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    check_point = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, save_weights_only=True)
    print('input shape', train_data.shape, train_lexical_feature.shape, train_pos.shape)
    model.fit([train_data, train_lexical_feature, train_pos], train_y, validation_split=0.1, epochs=50,
              batch_size=BATCH_SIZE,
              shuffle=True, callbacks=[early_stopping, check_point], class_weight={1: 3, 0: 1})


def evalute(model, test_data, test_local_feature, test_pos, test_y):
    model.load_weights(MODEL_SAVE_PATH)
    y_pred = model.predict([test_data, test_local_feature, test_pos], batch_size=300, verbose=1)
    # print ('predict', y_pred)
    print('success')

    correct_list, wrong_list, miss_list = softmax_evaluation(test_y, y_pred)

    '''
    print ('预测正确')
    for i in correct_list:
        print(test_x[i].replace(' ', ''), test_word[i])

    print()
    print ('预测错误')
    for i in wrong_list:
        print(test_x[i].replace(' ', ''), test_word[i])

    print()
    print ('未能召回')
    for i in miss_list:
        print(test_x[i].replace(' ', ''), test_word[i])

    '''

def evaluate_neg_by_sentence(model, test_data, test_local_feature, test_pos, test_x):
    """
    句子级别的评估，当有一个词有触发词是，则该句代表一个事件，直接判定句子是不是分类正确
    这个方法针对的是负样本的句子，所有的句子其标注都为非事件
    :param model:
    :param test_data:
    :param test_local_feature:
    :param test_pos:
    :return:
    """
    model.load_weights(MODEL_SAVE_PATH)
    y_pred = model.predict([test_data, test_local_feature, test_pos], batch_size=300, verbose=1)

    correct_set = set()
    total_set = set()
    for i in range(len(y_pred)):
        total_set.add(test_x[i])
        if np.argmax(y_pred[i]) == 0:
            correct_set.add(test_x[i])
    print ('负样本precision', (len(correct_set) / len(total_set)))
    return len(correct_set), len(total_set)

def evaluate_pos_by_sentence(model, test_data, test_local_feature, test_pos, test_x):
    """
    句子级别的评估，当有一个词有触发词是，则该句代表一个事件，直接判定句子是不是分类正确
    这个方法针对的是正样本的句子，所有的句子其标注都为事件
    :param model:
    :param test_data:
    :param test_local_feature:
    :param test_pos:
    :return:
    """
    model.load_weights(MODEL_SAVE_PATH)
    y_pred = model.predict([test_data, test_local_feature, test_pos], batch_size=300, verbose=1)


    correct_set = set()
    total_set = set()
    for i in range(len(y_pred)):
        total_set.add(test_x[i])
        if np.argmax(y_pred[i]) == 1:
            correct_set.add(test_x[i])
    print('正样本precision', len(correct_set) / len(total_set))

    return len(correct_set), len(total_set)

def calculate_fscore_by_sentence(pos_correct, pos_total, neg_correct, neg_total):
    """
    :param pos_correct: 正样本中分类正确的数量
    :param pos_total: 所有正样本的数量
    :param neg_correct: 负样本中分类正确的数量
    :param neg_total: 所有负样本的数量
    :return:
    """
    print ('pos_correct', pos_correct, 'pos_total', pos_total, 'neg_correct', neg_correct, 'neg_total', neg_total)
    precision = pos_correct * 1.0 / (pos_correct + neg_total - neg_correct)
    recall = pos_correct * 1.0 / pos_total
    f_score = 2 * precision * recall / (recall + precision)
    print ('句子级别的precision', precision, '句子级别的recall', recall, '句子级别的fscore', f_score)


if __name__ == '__main__':
        model, tokenizer = build_network()
        #train
        # train_data, train_lexical_feature, train_pos, train_y = prepare_train_data(tokenizer)
        # train(model, train_data, train_lexical_feature, train_pos, train_y)

        #正样本校验
        test_data, test_local_feature, test_pos, test_y, test_x = prepare_validate_data(tokenizer, 'word-multi-conv-val.txt')
        evalute(model, test_data, test_local_feature, test_pos, test_y)
        pos_correct, pos_total = evaluate_pos_by_sentence(model, test_data, test_local_feature, test_pos, test_x);
        #负样本校验
        test_data, test_local_feature, test_pos, test_y, test_x = prepare_validate_data(tokenizer, 'word-multi-conv-neg.txt')
        neg_correct, neg_total = evaluate_neg_by_sentence(model, test_data, test_local_feature, test_pos, test_x);

        #句子级别的评估
        calculate_fscore_by_sentence(pos_correct, pos_total, neg_correct, neg_total)
