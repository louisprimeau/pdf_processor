import pandas as pd


df = pd.read_csv('data.txt', sep='\t')

title_names = df['title']
journal_names = df['journal']


wiley_filter = [i for i, n in enumerate(list(journal_names)) if n[:8] == 'Physica ' or n[:7] == 'Adv.Mat']
wiley_title_names = df['title'][wiley_filter].dropna()

from elsapy.elsclient import ElsClient
from elsapy.elsprofile import ElsAuthor, ElsAffil
from elsapy.elsdoc import FullDoc, AbsDoc
from elsapy.elssearch import ElsSearch
import json
import requests
import re

with open("elsevier_api.key") as con_file:
    config = json.load(con_file)

    
APIKey = config['apikey']
client = ElsClient(APIKey)
headers = {
        'X-ELS-APIKEY': APIKey,
        'Accept': 'application/pdf',
        'view': 'FULL',
    }
_url_base= u'https://api.elsevier.com/content/article/'

keys = ['prism:doi', 'dc:title', 'prism:publicationName']
fields = []
for i, title in enumerate(wiley_title_names[:10]):
    print(i, title)
    clean_title = re.sub(r'[^A-Za-z0-9 -]+', '', title)
    search_query = 'TITLE-ABS-KEY ( {} )'.format(" AND ".join(clean_title.split(" ")[:10]))
    doc_srch = ElsSearch(search_query, 'scopus')
    doc_srch.execute(client, get_all = True)
    res_dict = doc_srch.results[0]
    if all(key in res_dict.keys() for key in keys):
        fields.append([res_dict['prism:doi'], res_dict['dc:title'], res_dict['prism:publicationName']])
    else:
        print("not found")

