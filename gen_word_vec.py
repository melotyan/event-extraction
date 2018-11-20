from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors
import jieba

def roll_back(path = 'words.txt', output = 'words-origin.txt'):
    with open(path, encoding='utf-8', errors='ignore') as f:
        res = f.read().replace(' ', '')
    with open(output, 'w') as f:
        f.write(res)


def cut_words(path = 'words-origin.txt', output = 'words2.txt'):
    res = ''
    with open(path) as f:
        for line in f:
            res += ' '.join(jieba.cut(line))
    with open(output, 'w') as f:
        f.write(res)


def train_word_embedding(path='words.txt', model_path = 'zh-word-vec100'):
    sentence = LineSentence(path)
    model = Word2Vec(sentences=sentence, size = 100, sg = 1)
    model.save(model_path)

    model.wv.save_word2vec_format('zh-word-vec100.txt', binary=False)
# cut_words()
train_word_embedding()