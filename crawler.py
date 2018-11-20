import requests

url = 'https://s3.amazonaws.com/arrival/embeddings/wiki.multi.en.vec'

res = requests.get(url)
print ('start')
with open('eng-vec.txt', 'w') as f:
    f.write(res.text)