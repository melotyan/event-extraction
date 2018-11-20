import numpy as np
from keras.preprocessing import text, sequence
from keras.layers import Embedding, Input, Dense, GlobalMaxPool1D, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from metrics import auc
from init_training_data import init_eng_wiki_trigger_data, init_eng_wiki_arg_data, init_word_safe_data
import web
import requests

MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
MAX_LEN = 80
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 32
WORD_TO_VECTOR_PATH = 'en.vec'
MODEL_SAVE_PATH = 'embedding-trigger2.h5'

def get_data():
    '''
    处理训练数据
    :return: 训练集，训练标签，校验集，校验标签，word-index
    '''
    #获取数据
    train_x, train_y = init_eng_wiki_trigger_data(max_len=MAX_LEN)
    train_y = np.asarray(train_y)
    test_x, test_y = train_x[-1000:], train_y[-1000:]
    train_x, train_y = train_x[:-1000], train_y[:-1000]
    print ('train_y shape', train_y.shape)

    #词-数
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(train_x + test_x)

    #给每个词标号
    train_sequence = tokenizer.texts_to_sequences(train_x)
    test_sequence = tokenizer.texts_to_sequences(test_x)
    word_index = tokenizer.word_index

    #补0，成为一个向量
    train_data = sequence.pad_sequences(sequences=train_sequence, maxlen=MAX_LEN, padding='post')
    test_data = sequence.pad_sequences(sequences=test_sequence, maxlen=MAX_LEN, padding='post')
    # train_y = sequence.pad_sequences(sequences=train_y, maxlen=MAX_LEN, padding='post')
    print ('train data shape', train_data.shape)

    return train_data, train_y, test_data, test_y, tokenizer, test_x

def get_embedding_vector(word_index):
    '''
    根据word_index返回训练数据对应的embedding matrix
    :param word_index:
    :return: 词个数，embedding矩阵
    '''
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
    print ('embedding shape', embeddings_matrix.shape)

    return nb_words, embeddings_matrix

def split_tain_val_data(train_data, train_y):
    '''
    将训练集划一部分出来，在训练的过程中做校验
    :param train_data:
    :param train_y:
    :return: 训练集，训练标签，检验集，检验标签
    '''
    #划分训练集和验证集
    perm = np.random.permutation(len(train_data))
    idx_train = perm[:int(len(train_data) * (1 - VALIDATION_SPLIT))]
    idx_val = perm[int(len(train_data) * (1 - VALIDATION_SPLIT)):]

    val_data = train_data[idx_val]
    val_label = train_y[idx_val]
    print ('validate data shape', val_data.shape, val_label.shape)

    train_data = train_data[idx_train]
    train_label = train_y[idx_train]
    print ('train data shape', train_data.shape, train_data.shape)

    return train_data, train_label, val_data, val_label


class MyModel:
    def __init__(self, weight_path = MODEL_SAVE_PATH, nb_words = None, embeddings_matrix = None):
        self.model = self.build_netword(nb_words, embeddings_matrix)
        self.weight_path = weight_path

    def build_netword(self, nb_words, embeddings_matrix):
        '''
        建网络模型
        :param nb_words:
        :param embeddings_matrix:
        :return:
        '''
        #建网络
        text_input = Input(shape=(MAX_LEN,), dtype='int32')
        embedding_sequence = Embedding(input_dim=nb_words, output_dim=EMBEDDING_DIM, weights=[embeddings_matrix],
                                    input_length=MAX_LEN, trainable=False)(text_input)
        x = Dense(100, activation='tanh')(embedding_sequence)
        x = Dense(100, activation='tanh')(x)
        x = Dense(100, activation='tanh')(x)
        x = Dense(100, activation='tanh')(x)
        # x = Dense(150, activation='tanh')(x)
        x = Flatten()(x)
        x = Dense(MAX_LEN, activation='sigmoid')(x)

        model = Model(inputs=[text_input], outputs=x)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc])
        model.summary()

        return model

    def train(self, train_data, train_label, val_data, val_label):
        earlystoping = EarlyStopping(monitor='val_loss', patience=5)
        # best_weight_path = MODEL_SAVE_PATH
        model_chekpoint = ModelCheckpoint(self.weight_path, save_best_only=True, save_weights_only=True)
        self.model.fit(train_data, train_label, validation_data=(val_data, val_label), shuffle=True,
                  batch_size=BATCH_SIZE, epochs=100, callbacks=[earlystoping, model_chekpoint])

    def load_weights(self):
        self.model.load_weights(self.weight_path)

    def predict(self, tokenizer, sentences):
        print ('type sentences', type(sentences))
        # tokenizer = text.Tokenizer()
        sequences = tokenizer.texts_to_sequences(sentences)
        # print ('sequences', sequences)
        data = sequence.pad_sequences(sequences=sequences, maxlen=MAX_LEN, padding='post')

        y_pred = self.model.predict(data, batch_size=30, verbose=1)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        for i in range(len(y_pred)):
            if sum(y_pred[i]) == 0:
                continue
            indexs = [j for j in range(len(y_pred[i])) if y_pred[i][j] == 1]
            print ('句子', sentences[i])
            sen_arr = sentences[i].split()
            print ('触发词', end=' ')
            for j in indexs:
                # if tokenizer.word_index[sen_arr[j]] == data[j]:
                print (sen_arr[j], end=' ')
                # else:
                #     #用split()分词可能与模型的分词结果不一致，最后下标相差一两个词
                #     for k in range(-2, 3):
                #         if j + k < len(sen_arr) and tokenizer.word_index[sen_arr[j + k]] == data[j + k]:
                #             print(sen_arr[j], end=' ')
                #             break

            print ()

    def evaluate(self, test_data, test_x, test_y):
        #测试
        y_pred = self.model.predict(test_data, batch_size=30, verbose=1)
        print ('predict', y_pred)
        # print (sum(y_pred[0]))
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        print ('success')

        count = 0
        target_count = 0
        index_list = []
        precision_count = 0

        for i in range(len(y_pred)):
            test_arr = test_x[i].split()
            if sum(y_pred[i]) >= 1: #检测出了触发词
                index = []
                print (test_x[i])
                # print (test_x[i].replace(' ', ''))
                #预测项
                print ('预测结果')
                for j in range(len(y_pred[i])):
                    if j < len(test_arr) and y_pred[i][j] == 1:
                        count += 1
                        print (test_arr[j], end=' ')
                        if test_y[i][j] == 1:
                            precision_count += 1
                        index.append(j)
                print ('')
                print ('真实结果')
                for j in range(len(test_y[i])):
                    if j < len(test_arr) and test_y[i][j] == 1:
                        print (test_arr[j], end=' ')
                        index.append(j)
                print('')

            #真实项
            if sum(test_y[i]) >= 1:
                for j in range(len(test_y[i])) :
                    if j < len(test_arr) and test_y[i][j] == 1:
                        target_count += 1

        pre = precision_count * 1.0 / count
        recall = precision_count * 1.0 / target_count
        print ('precision', pre, 'recall', recall, 'F score', 2 * pre * recall / (pre + recall))
        print ('precision count', precision_count, 'count is', count, 'target count is', target_count)
        print (index_list)


def load_my_model():
    train_data, train_y, test_data, test_y, tokenizer, test_x = get_data()
    nb_words, embedding_matrix = get_embedding_vector(tokenizer.word_index)
    train_data, train_y, val_data, val_y = split_tain_val_data(train_data, train_y)

    myModel = MyModel(weight_path=MODEL_SAVE_PATH, nb_words=nb_words, embeddings_matrix=embedding_matrix)
    return myModel, tokenizer

class events:
    def POST(self):
        data = web.input()
        print (data['sens'], type(data['sens']))
        myModel.predict(tokenizer, [data['sens']])


def test_events_api():
    url = 'http://0.0.0.0:8080/events'
    data = {'sens': ['Edward II was the fourth son of Edward I and his first wife, Eleanor of Castile.',
                     'Natalie is the winner of Miss Universe 2005 and also the wife of Thai tennis player Paradorn Srichaphan.']}
    res = requests.post(url=url, data=data)
    print (res)

if __name__ == '__main__':
    urls = (
        '/events/*', 'events'
    )
    myModel, tokenizer = load_my_model()
    myModel.load_weights()
    app = web.application(urls, globals(), autoreload=True)
    app.run()
    # test_events_api()

    # myModel.evaluate(test_data, test_x, test_y)
    # myModel.predict(tokenizer, ['Edward II was the fourth son of Edward I and his first wife, Eleanor of Castile.',
    #                  'Natalie is the winner of Miss Universe 2005 and also the wife of Thai tennis player Paradorn Srichaphan.',
    #                  'Born about 1320, Edward was the only son of Thomas of Brotherton, eldest son of Edward I by his second marriage to Margaret (1279?'])

