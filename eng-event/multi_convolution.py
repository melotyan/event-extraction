from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Dense, Input, Bidirectional, LSTM, Conv1D, Flatten, MaxPool1D
from keras.layers import GlobalMaxPool1D, Concatenate, Dropout, Reshape
from keras.layers.core import RepeatVector
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import RandomUniform
# from attention import Attention
import numpy as np
from metrics import auc, softmax_evaluation, sigmoid_evaluation
from init_training_data import get_multi_conv_data
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




#数据获取
words, train_x, train_lexical, train_pos, train_y = get_multi_conv_data('multi-conv.json')
words = words[0:200000]
train_x = train_x[0:200000]
train_lexical = train_lexical[0:200000]
train_pos = train_pos[0:200000]
train_y = train_y[0:200000]
# train_pos = np.asarray(train_pos)
#转成category格式
from keras.utils.np_utils import to_categorical
train_y = to_categorical(train_y, num_classes=2)

print ('总样本数', sum(np.argmax(a) for a in train_y))
# train_y = np.asarray(train_y)
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
# perm = np.random.permutation(len(train_data))
# idx_train = perm[:int(len(train_data) * (1 - VALIDATION_SPLIT))]
# idx_val = perm[int(len(train_data) * (1 - VALIDATION_SPLIT)):]
#
# val_data = train_data[idx_val]
# val_local = train_lexical_feature[idx_val]
# val_label = train_y[idx_val]
# print ('validate data shape', val_data.shape, val_local.shape, val_label.shape)
#
# train_data = train_data[idx_train]
# train_lexical_feature = train_lexical_feature[idx_train]
# train_label = train_y[idx_train]
# print ('train data shape', train_data.shape, train_lexical_feature.shape, train_label.shape)

#定义网络层
sentence_embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embeddings_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, trainable=False)
lexical_embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embeddings_matrix],
                            input_length=CONTEXT_SIZE, trainable=False)
position_embedding_layer = Embedding(nb_words, 10, input_length=MAX_SEQUENCE_LENGTH, trainable=True)


cnn_layer1 = Conv1D(filters=32, kernel_size=(3))#, kernel_initializer=RandomUniform(minval=-0.01, maxval=0.01))
#构建模型
sentence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='sentence_input')
lexical_input = Input(shape=(CONTEXT_SIZE,), dtype='int32', name='lexical_input')
position_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='position_input')

sentence_embedding_sequence = sentence_embedding_layer(sentence_input)
x2 = lexical_embedding_layer(lexical_input)

position_embedding_sequence = position_embedding_layer(position_input)

x1 = Concatenate(axis=-1)([sentence_embedding_sequence, position_embedding_sequence, position_embedding_sequence])
print ('x1 shape', x1.shape)
x1 = cnn_layer1(x1)
x1 = GlobalMaxPool1D()(x1)
x2 = Reshape((3 * EMBEDDING_DIM,))(x2)
print ('x2 x1 shape', x2.shape, x1.shape)
x = Concatenate(axis=-1)([x1, x2])
print('x shape', x.shape)
# x = Dense(100, activation='relu')(x)
# x = Dense(100, activation='relu')(x)
# x = Dense(100, activation='relu')(x)
# x = Dense(100, activation='relu')(x)
# x = Dense(100, activation='relu')(x)
# x = Dense(100, activation='relu')(x)
# x = Dense(100, activation='relu')(x)
# x = Dense(100, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dense(100, activation='relu')(x)

x = Dropout(0.5)(x)
x = Dense(2, activation='softmax', name='output')(x)

model = Model(inputs=[sentence_input, lexical_input, position_input], outputs=x)
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=[auc])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
best_weigth_path = MODEL_SAVE_PATH
check_point = ModelCheckpoint(best_weigth_path, save_best_only=True, save_weights_only=True)
print ('output x shape', x.shape)
print('input shape', train_data.shape, train_lexical_feature.shape, train_pos.shape)
model.fit([train_data, train_lexical_feature, train_pos], train_y, validation_split = 0.1, epochs=50, batch_size=BATCH_SIZE,
          shuffle=True, callbacks=[early_stopping, check_point], class_weight={1:15, 0:1})#,
# model.save('con-bilstm.h5')
model.load_weights(best_weigth_path)

#测试
y_pred = model.predict([test_data, test_local_feature, test_pos], batch_size=300, verbose=1)
# print ('predict', y_pred)
print ('success')

softmax_evaluation(test_y, y_pred)




