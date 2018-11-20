import json
import numpy as np
import jieba


MAX_LEN = 80

def init_char_traffic_data(path='train-data.txt'):
    '''
    按字符划分
    :param path:
    :return:
    '''
    train = []
    label = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            sen = data['sentence']
            denotors = data['denotors']
            temp = np.zeros(MAX_LEN)
            for denotor in denotors:
                index = sen.find(denotor)
                temp[index:len(denotor) + index] = 1
            sen = ' '.join(sen)
            train.append(sen)
            label.append(temp)
    # total 2208 rows,
    return train, label

def init_word_traffic_data(path='train-data.txt'):
    '''
    按词语划分，不按字符划分
    :param path:
    :return:
    '''
    train = []
    label = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            denotors = data['denotors']
            temp = np.zeros(MAX_LEN)
            sen_arr = list(jieba.cut(data['sentence']))

            for denotor in denotors:
                for word in sen_arr:
                    if (denotor in word or word in denotor) and sen_arr.index(word) < MAX_LEN:
                        temp[sen_arr.index(word)] = 1
            sen = ' '.join(sen_arr)
            train.append(sen)
            label.append(temp)
    # total 2208 rows,
    return train, label

def init_char_safe_data(path='safe-event.txt', max_len=MAX_LEN):
    train = []
    label = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            sen = data['sentence']
            trigger = data['trigger']

            temp = np.zeros(max_len)
            index = sen.find(trigger)
            # temp[index] = 1 #softmax多分类需要用，1表示trigger开头，2表示trigger后面部分
            temp[index:len(trigger) + index] = 1
            sen = ' '.join(sen)
            train.append(sen)
            label.append(temp)

    return train, label

def init_word_safe_data(path='safe-event.txt', max_len=MAX_LEN):
    train, label = [], []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            # sen = data['sentence']
            trigger = data['trigger']
            temp = np.zeros(max_len)
            sen_arr = list(jieba.cut(data['sentence']))

            for word in sen_arr:
                if (word in trigger or trigger in word) and sen_arr.index(word) < max_len:
                    temp[sen_arr.index(word)] = 1
            if sum(temp) == 0:
                continue
            sen = ' '.join(sen_arr)
            train.append(sen)
            label.append(temp)

    return train, label

def init_eng_wiki_trigger_data(path1='wiki_sentence.txt', path2 = 'eng-event/fn_trigger.txt', max_len = 80):
    train, label = [], []

    # with open(path2) as f:
    #     for line in f:
    #         data = json.loads(line)
    #         sen = data['sen']
    #         trigger = data['trigger']
    #         sen_arr = sen.split()
    #         trigger_arr = trigger.split()
    #
    #         temp = np.zeros(max_len)
    #         for t in trigger_arr:
    #             for s in sen_arr:
    #                 if t.lower() == s.lower() and sen_arr.index(s) < max_len:
    #                     temp[sen_arr.index(s)] = 1
    #
    #         train.append(sen)
    #         label.append(temp)

    with open(path1) as f:
        for line in f:
            data = json.loads(line)
            sen = data['sen']
            trigger = data['trigger']
            sen_arr = sen.split()
            trigger_arr = trigger.split()

            temp = np.zeros(max_len)
            for t in trigger_arr:
                for s in sen_arr:
                    if t.lower() == s.lower() and sen_arr.index(s) < max_len:
                        temp[sen_arr.index(s)] = 1
            if (sum(temp) == 0):
                continue
            train.append(sen)
            label.append(temp)

    return train, label

def init_eng_wiki_arg_data(path='wiki_sentence.txt', max_len = 80):
    train, label = [], []

    with open(path) as f:
        lines = f.readlines()
        np.random.shuffle(lines)
        for line in lines:
            data = json.loads(line)
            sen = data['sen']
            args = data['args']
            sen_arr = sen.split()

            temp = np.zeros(max_len)
            for arg in args:
                arg_arr = arg.split()
                for t in arg_arr:
                    for s in sen_arr:
                        if t.lower() == s.lower() and sen_arr.index(s) < max_len:
                            temp[sen_arr.index(s)] = 1

            train.append(sen)
            label.append(temp)

    return train, label

def init_test_word_safe_data(path1 = 'safe-event.txt', path2 = 'safe-event2.txt', max_len=MAX_LEN):
    train = []
    with open(path2) as f:
        for line in f:
            data = json.loads(line)
            # sen = data['sentence']
            trigger = data['trigger']
            temp = np.zeros(max_len)
            sen_arr = list(jieba.cut(data['sentence']))
            sen = ' '.join(sen_arr)
            train.append(sen)

    test, label = [], []
    with open(path1) as f:
        for line in f:
            data = json.loads(line)
            trigger = data['trigger']
            temp = np.zeros(max_len)
            sen_arr = list(jieba.cut(data['sentence']))

            for word in sen_arr:
                if (word in trigger or trigger in word) and sen_arr.index(word) < max_len:
                    temp[sen_arr.index(word)] = 1
            sen = ' '.join(sen_arr)
            if sen not in train:
                test.append(sen)
                label.append(temp)
    return test, label

def gen_eng_wiki_to_word_format(path='eng-event/wiki_sentence.txt', max_len = 80):
    '''
    把英文的每个词折开，拿它的sentence feature 和 lexical feature进行组合，做成新的输入
    :param path:
    :param max_len:
    :return:
    '''
    res = []

    with open(path) as f:
        for line in f:
            data = json.loads(line)
            sen = data['sen']
            sen_arr = sen.split()[0:max_len]
            trigger = data['trigger'].lower()
            trigger_arr = trigger.split()

            temp = []
            has_trigger = False
            for s in sen_arr:
                index = sen_arr.index(s)
                local = sen_arr[max(0, index - 3): index + 4]
                if local.index(s) < 3:
                    local = ['P'] * (3 - local.index(s)) + local
                if len(local) < 7:
                    local = local + ['P'] * (7 - len(local))
                local = ' '.join(local)
                label = 0
                if s.lower() in trigger_arr:
                    has_trigger = True
                    label = 1 #if trigger_arr.index(s.lower()) == 0 else 2
                temp.append(s + '\t' + sen + '\t' + local + '\t' + str(label) + '\n')
            if has_trigger:
                res.extend(temp)

    with open('eng-event/wiki_sentence_word_format.txt', 'w') as f:
        f.writelines(res)

def init_format_data(path = 'wiki_sentence_word_format.txt'):
    '''
    直接读取处理好的数据，方便模型按字进行输入
    :param path:
    :return:
    '''
    words = []
    sens = []
    locals = []
    labels = []
    with open(path) as f:
        for line in f:
            arr = line.strip().split('\t')
            if (len(arr) != 4):
                continue
            words.append(arr[0])
            sens.append(arr[1])
            locals.append(arr[2])
            labels.append(arr[3])

    return words, sens, locals, labels

def gen_word_safe_format_data(path='safe-event/safe-event.txt', max_len = 50):
    '''
    把中文安全事件训练语料中的每个词拆开，结合它的句子特征和词法特征，整合输入
    :param path:
    :return:
    '''
    res = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            # sen = data['sentence']
            trigger = data['trigger']
            temp = np.zeros(max_len)
            sen_arr = list(jieba.cut(data['sentence']))

            if trigger not in sen_arr or sen_arr.index(trigger) >= max_len: #分词有问题 或 长度超了
                continue

            for s in sen_arr:
                index = sen_arr.index(s)
                if (index == -1):
                    continue
                local = sen_arr[max(0, index - 3): index + 4]
                if local.index(s) < 3:
                    local = ['P'] * (3 - local.index(s)) + local
                if len(local) < 7:
                    local = local + ['P'] * (7 - len(local))
                local = ' '.join(local)
                label = 1 if s == trigger else 0

                res.append(s + '\t' + ' '.join(sen_arr) + '\t' + local + '\t' + str(label) + '\n')

    with open('safe-event/word-format.txt', 'w') as f:
        f.writelines(res)

# gen_eng_wiki_to_word_format()

# def gen_word_safe_multi_con_data(path='safe-event/safe-event.txt', max_len = 50):
