from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Dense, Input, Bidirectional, LSTM, Conv1D, Flatten
from keras.layers import GlobalMaxPool1D, Concatenate, Dropout, Activation, BatchNormalization
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import RandomUniform
from attention import Attention
import numpy as np
from metrics import auc
from init_training_data import init_eng_wiki_trigger_data, init_eng_wiki_arg_data


vocabulary_size = 20000
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
MAX_SEQUENCE_LENGTH = 80
BATCH_SIZE = 32
WORD_TO_VECTOR_PATH = 'eng-event/en.vec'
MODEL_SAVE_PATH = 'eng-event/con-bilstm-arg.h5'
num_lstm = 128




#数据获取
train_x, train_y = init_eng_wiki_arg_data(max_len=MAX_SEQUENCE_LENGTH)
train_y = np.asarray(train_y)
test_x, test_y = train_x[-1000:], train_y[-1000:]
train_x, train_y = train_x[:-1000], train_y[:-1000]
print ('train_y shape', train_y.shape)

#词-数
tokenizer = text.Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(train_x + test_x)
word_index = tokenizer.word_index

#生成数字序列
train_sequence = tokenizer.texts_to_sequences(train_x)
test_sequence = tokenizer.texts_to_sequences(test_x)
#补齐
train_data = pad_sequences(sequences=train_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
test_data = pad_sequences(sequences=test_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
print ('train_data shape', train_data.shape, 'test data shape', test_data.shape)

# 测试
# model = load_model('model.h5')
# pred = model.predict(train_data[:10])
# print ('sum', sum(pred[0]))
# pred[pred >= 0.5] = 1
# pred[pred < 0.5] = 0
# for i in range(len(pred)):
#     if sum(pred[i]) > 0:
#         print (pred)
# exit(0)

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

#划分训练集和测试集
perm = np.random.permutation(len(train_data))
idx_train = perm[:int(len(train_data) * (1 - VALIDATION_SPLIT))]
idx_val = perm[int(len(train_data) * (1 - VALIDATION_SPLIT)):]

val_data = train_data[idx_val]
val_label = train_y[idx_val]
print ('validate data shape', val_data.shape, val_label.shape)

train_data = train_data[idx_train]
train_label = train_y[idx_train]
print ('train data shape', train_data.shape, train_data.shape)

#定义网络层
embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embeddings_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, trainable=False)
bi_lstm_layer = Bidirectional(LSTM(num_lstm, kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=573210102),
                                   return_sequences=True, dropout=0.5))
cnn_layer = Conv1D(filters=32, kernel_size=(7), kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=729230112))

#构建模型
inp = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_sequence = embedding_layer(inp)
x1 = bi_lstm_layer(embedding_sequence)
# x1 = Attention(MAX_SEQUENCE_LENGTH)(x1)
x1 = Flatten()(x1)
x2 = cnn_layer(embedding_sequence)
x2 = GlobalMaxPool1D()(x2)
x = Concatenate()([x1, x2])
x = Dense(MAX_SEQUENCE_LENGTH)(x)
x = Dense(MAX_SEQUENCE_LENGTH, activation='sigmoid')(x)

model = Model(inputs=inp, outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
best_weigth_path = MODEL_SAVE_PATH
check_point = ModelCheckpoint(best_weigth_path, save_best_only=True, save_weights_only=True)
# model.fit(train_data, train_label, validation_data=(val_data, val_label), epochs=50, batch_size=BATCH_SIZE,
#           shuffle=True, callbacks=[early_stopping, check_point]) #, class_weight={1:0.9, 0:0.1})
# model.save('con-bilstm.h5')
model.load_weights(best_weigth_path)

#测试
y_pred = model.predict(test_data, batch_size=30, verbose=1)
print ('predict', y_pred)
# print (sum(y_pred[0]))
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0
print ('success')

count = 0
target_count = 0
precision_count = 0
index_list = []


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
                print (test_arr[j])
                if test_y[i][j] == 1:
                    precision_count += 1
                index.append(j)
        # print ('')
        print ('真实结果')
        for j in range(len(test_y[i])):
            if j < len(test_arr) and test_y[i][j] == 1:
                print (test_arr[j])
                index.append(j)
        # print('')

    #真实项
    if sum(test_y[i]) >= 1:
        for j in range(len(test_y[i])) :
            if j < len(test_arr) and test_y[i][j] == 1:
                target_count += 1

print ('precision', precision_count * 1.0 / count, 'recall', precision_count * 1.0 / target_count)
print ('precision count', precision_count, 'count is', count, 'target count is', target_count)
print (index_list)

def init_wordvec():
    #读取预训练的词向量
    embeddings_index = {}
    with open('vec.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vec
    print('Total %s word vectors.' % len(embeddings_index))

    #提取需要的词向量矩阵
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embeddings_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector:
            embeddings_matrix[i] = embedding_vector

    return embeddings_matrix




