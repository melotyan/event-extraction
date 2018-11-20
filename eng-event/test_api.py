import requests
import json

url = 'http://0.0.0.0:5000/events/'
list = []
# list.append('Edward II was the fourth son of Edward I and his first wife, Eleanor of Castile.')
# list.append('Natalie is the winner of Miss Universe 2005 and also the wife of Thai tennis player Paradorn Srichaphan.')
data = {'sens': ['Edward II was the fourth son of Edward I and his first wife, Eleanor of Castile.',
                 'Natalie is the winner of Miss Universe 2005 and also the wife of Thai tennis player Paradorn Srichaphan.',
                 'Born about 1320, Edward was the only son of Thomas of Brotherton, eldest son of Edward I by his second marriage to Margaret (1279?']}
# data = {'sens':list}
res = requests.post(url=url, data=data)
print (res.text)

