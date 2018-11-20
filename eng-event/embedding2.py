import numpy as np
from keras.preprocessing import text, sequence
from keras.layers import Embedding, Input, Dense, GlobalMaxPool1D, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from metrics import auc
from init_training_data import init_eng_wiki_trigger_data

MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
MAX_LEN = 80
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 32
WORD_TO_VECTOR_PATH = 'en.vec'
MODEL_SAVE_PATH = 'con-bilstm-trigger.h5'


#获取数据
train_x, train_y = init_eng_wiki_trigger_data(max_len=MAX_LEN)
#转成category格式
from keras.utils.np_utils import to_categorical
train_y = to_categorical(train_y, num_classes=2)
temp_y = []
for i in range(len(train_y)):
    temp_y.append(train_y[i].flatten())
train_y = np.asarray(temp_y)

test_x, test_y = train_x[-1000:], train_y[-1000:]
train_x, train_y = train_x[:-1000], train_y[:-1000]
print ('train_y shape', train_y.shape)
print (train_x[2])
print (train_y[2])

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

#建网络
text_input = Input(shape=(MAX_LEN,), dtype='int32')
embedding_sequence = Embedding(input_dim=nb_words, output_dim=EMBEDDING_DIM, weights=[embeddings_matrix],
                            input_length=MAX_LEN, trainable=False)(text_input)
x = Flatten()(embedding_sequence)
x = Dense(100, activation='tanh')(x)
# x = Flatten()(x)
# x = Dense(100, activation='tanh')(x)
x = Dense(MAX_LEN * 2)(x)
# x = Dense(3, activation='softmax')(x)

model = Model(inputs=[text_input], outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc])
model.summary()

earlystoping = EarlyStopping(monitor='val_loss', patience=5)
best_weight_path = MODEL_SAVE_PATH
model_chekpoint = ModelCheckpoint(best_weight_path, save_best_only=True, save_weights_only=True)
model.fit(train_data, train_label, validation_data=(val_data, val_label), shuffle=True,
          batch_size=BATCH_SIZE, epochs=100, callbacks=[earlystoping, model_chekpoint])
model.load_weights(best_weight_path)

#测试
y_pred = model.predict(test_data, batch_size=30, verbose=1)
print ('predict', y_pred)
print (sum(y_pred[0]))
# y_pred[y_pred >= 0.1] = 1
# y_pred[y_pred < 0.1] = 0
print ('success')

count = 0
target_count = 0
index_list = []
precision_count = 0

for i in range(len(y_pred)):
    print (test_x[i])
    test_arr = test_x[i].split()

    pred_temp = y_pred[i].reshape(MAX_LEN, 2)
    true_temp = test_y[i].reshape(MAX_LEN, 2)

    for k in range(min(len(test_arr), len(pred_temp))): #判断每一个字符
        pred_max_indexs = np.where(pred_temp[k]==np.max(pred_temp[k]))
        true_max_indexs = np.where(true_temp[k] == np.max(true_temp[k]))
        # print ('预测结果')
        if pred_max_indexs[0] == 1:#检测出了触发词的开始
            count += 1
            print ('预测结果')
            print (test_arr[k], end='')
            if true_max_indexs[0] != 0:
                y_pred += 1

        if true_max_indexs[0] == 1:
            # print('真实结果', true_max_indexs)
            target_count += 1
            print (test_arr[k], end='')


    print()

print ('precision count', precision_count, 'count is', count, 'target count is', target_count)
print ('precision', precision_count * 1.0 / count, 'recall', precision_count * 1.0 / target_count)
print (index_list)




def init_wordvec():
    #读取预训练的词向量
    word_index = {}
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