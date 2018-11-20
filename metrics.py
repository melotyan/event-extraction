from keras import backend as K
import tensorflow as tf
import numpy as np


def crossentry_with_weight(labels, pred):
    K.binary_crossentropy()


def precision(labels, pred):
    print ('type labels', labels, labels.shape)
    index_labels = [i for i in range(labels.shape[0]) if labels[i] == 1]
    count = 0
    for index in index_labels:
        if pred[index] == 1:
            count += 1

    return count * 1.0 / len(index_labels)

# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)


def softmax_evaluation(y_true, y_pred):
    positive_count = sum(np.argmax(a) for a in y_true)  # 测试样例中正例的个数
    tp_count = 0  # true positive的个数，实际和预测都为1
    fp_count = 0 # false positive的个数， 实际为0， 预测为1
    correct_count = 0  # 预测正确的个数

    correct_list = []
    wrong_list = []
    miss_list = []

    for i in range(len(y_pred)):

        if np.argmax(y_pred[i]) == np.argmax(y_true[i]):  # 预测正确
            correct_count += 1
            if np.argmax(y_pred[i]) == 1:  # 预测和目标都显示是触发词
                tp_count += 1
                correct_list.append(i)
        else:
            if np.argmax(y_pred[i]) == 1: #预测为触发词，实际不是
                fp_count += 1
                wrong_list.append(i)
            else: #实际是触发词，没预测到
                miss_list.append(i)

    print('总样本数为', len(y_pred), '正样本数为', positive_count, '预测正确数为', correct_count, 'true positive', tp_count, 'false positive', fp_count)
    accuracy = correct_count * 1.0 / len(y_pred)
    precision = tp_count * 1.0 / (tp_count + fp_count)
    recall = tp_count * 1.0 / positive_count
    f_score = 2 * precision * recall / (recall + precision)
    print('accuracy', accuracy, 'precision', precision, 'recall', recall, 'F score', f_score)

    return correct_list, wrong_list, miss_list

def sigmoid_evaluation(y_true, y_pred):
    correct_count = 0
    true_positive = 0
    false_positive = 0
    positive_count = sum(int(i) for i in y_true)

    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    for i in range(len(y_true)):
        if int(y_true[i]) == y_pred[i]: #预测正确
            correct_count += 1
            if y_pred[i] == 1: #预测和实际都为1
                true_positive += 1
        else: #预测错误
            if y_pred[i] == 0: #实际为0，预测为1
                false_positive += 1

    print ('总样本数', len(y_true), '正样本数', positive_count, '分类正确数', correct_count,
           'true positive', true_positive, 'false_positive', false_positive)
    accuracy = correct_count * 1.0 / len(y_true)
    precision = true_positive * 1.0 / (true_positive + false_positive)
    recall = true_positive * 1.0 / positive_count
    f_score = precision * recall * 2 / (precision + recall)

    print ('accuracy', accuracy, 'precision', precision, 'recall', recall, 'f score', f_score)