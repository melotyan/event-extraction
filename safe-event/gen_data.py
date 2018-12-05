import xlwt
import xlrd
import pandas as pd
import json
import numpy as np
import re
import jieba

def extract_from_excel():
    res = []
    work_book = xlrd.open_workbook('ict.xlsx')
    sheet = work_book.sheet_by_name(work_book.sheet_names()[0])
    for i in range(sheet.nrows):
        text = sheet.cell_value(i, 3).strip()
        if '触发词' in text:
            res.append(text + '\n')

    with open('ict.txt', 'w') as f:
        f.writelines(res)

    res = []
    with open('ict.txt') as f:
        for line in f:
            if line.strip() != '':
                res.append(line.lstrip())

    with open('ict.txt', 'w') as f:
        f.writelines(res)

def gen_json_from_txt():
    res = []
    data = {}
    with open('ict.txt') as f:
        for line in f:
            if '触发词' not in line:
                data['sentence'] = line.strip()
            else:
                trigger = re.findall('触发词：(.*?)，', line, re.S)[0]
                if trigger != '':
                    try:
                        data['trigger'] = trigger
                        data['param1'] = re.findall('参数1：(.*?)，', line, re.S)[0]
                        data['param2'] = re.findall('参数2：(.*?)}', line, re.S)[0]
                        res.append(json.dumps(data) + '\n')
                    except:
                        print ('ict.txt', line)

    with open('ict3.txt') as f:
        for line in f:
            if '触发词' not in line:
                data['sentence'] = line.replace('原句子：', '').replace('原句子:', '').strip()

            else:
                trigger = re.findall('触发词:(.*?),', line, re.S)[0]
                data['trigger'] = trigger
                if trigger != '':
                    try:
                        data['param1'] = re.findall('参数1:(.*?)参数2', line, re.S)[0]
                        data['param1'] = data['param1'].replace(',', '').replace('，', '')
                        data['param2'] = re.findall('参数2.(.*?)}', line, re.S)[0]
                        res.append(json.dumps(data) + '\n')
                    except Exception as e:
                        print('ict3.txt', line, e)

    np.random.shuffle(res)
    with open('safe-event.txt', 'w') as f:
        f.writelines(res)

def gen_neg_json_from_txt(max_len = 50):
    res = []
    with open('negative-sample.txt') as f:
        for line in f:
            data = json.loads(line)
            title = re.sub('\s+', '', data['title'])
            print (title)
            sen_arr = list(jieba.cut(title))
            for s in sen_arr:
                index = sen_arr.index(s)
                lexical = sen_arr[max(0, index - 1): index + 2]
                if lexical.index(s) < 1:
                    lexical = ['0'] + lexical
                if len(lexical) < 3:
                    lexical = lexical + ['0'] * (3 - len(lexical))

                lexical = ' '.join(lexical)
                label = 0

                data = {}
                # index = [str(index)] * max_len
                position = np.zeros(len(sen_arr))
                for i in range(len(sen_arr)):
                    position[i] = i - index
                data['candidate'] = s
                data['sen'] = ' '.join(sen_arr)
                data['lexical'] = lexical
                data['position'] = position.tolist()
                data['label'] = label
                res.append(json.dumps(data, ensure_ascii=False) + '\n')
    with open('word-multi-conv-neg.txt', 'w') as f:
        f.writelines(res)



def test_load_safe_event():
    with open('safe-event.txt') as f:
        for line in f:
            data = json.loads(line)
            print (data)


gen_neg_json_from_txt()
# gen_json_from_txt()
# test_load_safe_event()