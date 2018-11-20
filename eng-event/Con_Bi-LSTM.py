from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Dense, Input, Bidirectional, LSTM, Conv1D, Flatten, MaxPool1D
from keras.layers import GlobalMaxPool1D, Concatenate, Dropout
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import RandomUniform
from attention import Attention
import numpy as np
from metrics import auc, softmax_evaluation
from init_training_data import init_format_data


vocabulary_size = 20000
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
MAX_SEQUENCE_LENGTH = 80
CONTEXT_SIZE = 7
BATCH_SIZE = 32
WORD_TO_VECTOR_PATH = 'en.vec'
MODEL_SAVE_PATH = 'con-bilstm-trigger.h5'
num_lstm = 100




#数据获取
words, train_x, train_local, train_y = init_format_data()
# words = words[:200000]
train_x = train_x[:200000]
train_local = train_local[:200000]
train_y = train_y[:200000]
#转成category格式
from keras.utils.np_utils import to_categorical
train_y = to_categorical(train_y, num_classes=2)
print ('总样本数', sum(np.argmax(a) for a in train_y))
test_x, test_local, test_y = train_x[-20000:], train_local[-20000:], train_y[-20000:]
print ('测试样本数', sum(np.argmax(a) for a in test_y))
train_x, train_local, train_y = train_x[:-20000], train_local[:-20000], train_y[:-20000]
print ('train_y shape', train_y.shape)


#词-数
tokenizer = text.Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(words)
word_index = tokenizer.word_index

#生成数字序列
train_sequence = tokenizer.texts_to_sequences(train_x)
train_local_sequence = tokenizer.texts_to_sequences(train_local)
test_sequence = tokenizer.texts_to_sequences(test_x)
test_local_sequence = tokenizer.texts_to_sequences(test_local)

#补齐
train_data = pad_sequences(sequences=train_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
train_local_feature = pad_sequences(sequences=train_local_sequence, maxlen=CONTEXT_SIZE, padding='post')
test_data = pad_sequences(sequences=test_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
test_local_feature = pad_sequences(sequences = test_local_sequence, maxlen=CONTEXT_SIZE, padding='post')
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
val_local = train_local_feature[idx_val]
val_label = train_y[idx_val]
print ('validate data shape', val_data.shape, val_local.shape, val_label.shape)

train_data = train_data[idx_train]
train_local_feature = train_local_feature[idx_train]
train_label = train_y[idx_train]
print ('train data shape', train_data.shape, train_local_feature.shape, train_label.shape)

#定义网络层
sentence_embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embeddings_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, trainable=False)
lexical_embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embeddings_matrix],
                            input_length=CONTEXT_SIZE, trainable=False)
bi_lstm_layer = Bidirectional(LSTM(num_lstm))#, kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=573210102),
                                   #return_sequences=True, dropout=0.5))
cnn_layer = Conv1D(filters=32, kernel_size=(4))#, kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01, seed=729230112))

#构建模型
sentence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='sentence_input')
lexical_input = Input(shape=(CONTEXT_SIZE,), dtype='int32', name='lexical_input')

sentence_embedding_sequence = sentence_embedding_layer(sentence_input)
lexical_embedding_sequence = lexical_embedding_layer(lexical_input)

x1 = bi_lstm_layer(sentence_embedding_sequence)
x2 = cnn_layer(lexical_embedding_sequence)
print (x2.shape, x1.shape)
x2 = GlobalMaxPool1D()(x2)
print (x2.shape, x1.shape)
# x1 = Attention(MAX_SEQUENCE_LENGTH)(x1)
# x1 = Flatten()(x1)
# x2 = cnn_layer(embedding_sequence)
# x2 = GlobalMaxPool1D()(x2)
x = Concatenate()([x1, x2])
x = Dense(100, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dense(100, activation='relu')(x)
# x = Dense(100, activation='tanh')(x)
x = Dense(2, activation='softmax', name='output')(x)

model = Model(inputs=[sentence_input, lexical_input], outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[auc])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
best_weigth_path = MODEL_SAVE_PATH
check_point = ModelCheckpoint(best_weigth_path, save_best_only=True, save_weights_only=True)
print ('output x shape', x.shape)
model.fit({'sentence_input':train_data, 'lexical_input':train_local_feature}, {'output':train_label}, validation_data=([val_data, val_local], val_label), validation_split = 0.1, epochs=50, batch_size=BATCH_SIZE,
          shuffle=True, callbacks=[early_stopping, check_point], class_weight={1:0.9, 0:0.1})#,
# model.save('con-bilstm.h5')
model.load_weights(best_weigth_path)

#测试
y_pred = model.predict([test_data, test_local_feature], batch_size=32, verbose=1)

softmax_evaluation(test_y, y_pred)

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




