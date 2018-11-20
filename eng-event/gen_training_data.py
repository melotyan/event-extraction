import json
import re
import numpy as np

def event_id2event_type(path = '/Users/melot/Downloads/event-extraction/event_instance.tsv'):
    id_type = {}
    with open(path) as f:
        for line in f:
            arr = line.split()
            id_type[arr[0]] = arr[1]

    return id_type

def read_all_event_type(path = '/Users/melot/Downloads/event-extraction/event_instance.tsv'):
    res = set()
    with open(path) as f:
        for line in f:
            arr = line.split()
            res.add(arr[1])

    with open('event_types.txt', 'w') as f:
        i = 1
        f.write('none' + '\t' + '0' + '\n')
        for r in res:
            f.write(r + '\t' + str(i) + '\n')
            i += 1


def gen_data_from_tsv(input='wiki_sentence.tsv', output='wiki_sentence.txt'):
    id_type = event_id2event_type()
    res = []
    with open(input) as f:
        for line in f:
            arr = line.split('\t')
            sen = arr[3]
            event_id = arr[2]

            trigger = arr[-1]
            if 'trigger,' not in trigger:
                continue
            trigger = trigger.split('trigger,')[1]
            trigger = trigger.split(',')[0]

            args = []
            for arg in arr[4:-1]:
                index = arg.split(',')
                args.append(sen[int(index[-2]):int(index[-1])])

            data = {}
            data['sen'] = sen
            data['trigger'] = trigger
            data['args'] = args
            data['type'] = id_type[event_id]
            res.append(json.dumps(data) + '\n')
            # print (trigger, args, sen)

    print (json.loads(res[0]))
    np.random.shuffle(res)
    print (json.loads(res[0]))
    print (len(res))
    with open(output, 'w') as f:
        f.writelines(res)

def gen_data_from_txt(input='fn_event_result.txt', output='fn_trigger.txt'):
    res = []
    with open(input) as f:
        for line in f:
            if '##' in line:
                data = {}
                trigger = re.search('(\[\[.*?\]\])', line, re.S)
                if trigger:
                    trigger = trigger.group(0)

                head = re.search('(##\d+\s+)', line, re.S)
                if head:
                    head = head.group(0)

                data['trigger'] = trigger.replace('[[', '').replace(']]', '')
                data['sen'] = line.replace(head, '').replace('[[', '').replace(']]', ' ').rstrip()
                res.append(json.dumps(data) + '\n')

    print(json.loads(res[0]))
    np.random.shuffle(res)
    print(json.loads(res[0]))
    print(len(res))
    with open(output, 'w') as f:
        f.writelines(res)


read_all_event_type()
