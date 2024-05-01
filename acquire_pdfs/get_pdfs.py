from elsapy.elsclient import ElsClient
from elsapy.elsprofile import ElsAuthor, ElsAffil
from elsapy.elsdoc import FullDoc, AbsDoc
from elsapy.elssearch import ElsSearch
import json
import requests

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

#doc_srch = ElsSearch("PUBYEAR > 2010 AND SUBJAREA(PHYS) AND KEY(superconductivity experimental) AND DOCTYPE(ar)",'scopus') # scopus format
doc_srch = ElsSearch('TITLE-ABS-KEY ( superconductivity AND experimental ) AND ( LIMIT-TO ( SUBJAREA , "PHYS" ) ) AND ( LIMIT-TO ( DOCTYPE , "ar" ) ) AND ( LIMIT-TO ( LANGUAGE , "English" ) )', 'scopus')
doc_srch.execute(client, get_all = True)
print("total results: ", doc_srch.tot_num_res)
print("doc_srch has", len(doc_srch.results), "results.")


keys = ['prism:doi', 'dc:title', 'prism:publicationName']
fields = []
for res_dict in doc_srch.results:
    if all(key in res_dict.keys() for key in keys):
        fields.append([res_dict['prism:doi'], res_dict['dc:title'], res_dict['prism:publicationName']])

print("returned", len(fields), "dois.")

with open('dois.txt', 'a') as f:
    for doi, title, _ in fields:
        f.write('{},{}\n'.format(doi, title))
   
# .lower().replace(' ', '_') + '.pdf'
"""
url = _url_base + 'doi/' + str(doi)
res = requests.get(url, headers = headers)
if res.status_code == 200:
    with open(os.path.join(write_directory, file_name),'wb') as f:
        f.write(res.content)

"""
