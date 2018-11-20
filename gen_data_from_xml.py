import re
import os
import json


def gen_sentence(path='CEC/'):
    res = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filename = os.path.join(root, file)
            print (filename)
            with open(filename) as f:
                data = f.read()

            sens = re.findall('<Sentence>(.*?)</Sentence>', data, re.S)
            for sen in sens:
                denotors = re.findall('<Denoter.*?>(.*?)</Denoter>', sen, re.S)
                locations = re.findall('<Location.*?>(.*?)</Location>', sen, re.S)
                times = re.findall('<Time.*?>(.*?)</Time>', sen, re.S)
                sen = re.sub('<.*?>', '', sen)
                sen = sen.replace('\n', '').replace('\t', '')

                json_data = {}
                json_data['sentence'] = sen
                json_data['denotors'] = denotors
                json_data['times'] = times
                json_data['locations'] = locations
                res.append(json.dumps(json_data) + '\n')
            with open('train-data.txt', 'w') as f:
                f.writelines(res)


def load_data(path='train-data.txt'):
    with open(path) as f:
        for line in f:
            json_data = json.loads(line)
            denotors = json_data['denotors']
            locations = json_data['locations']
            times = json_data['times']
            sen = json_data['sentence']
            print (denotors)
            break

load_data()