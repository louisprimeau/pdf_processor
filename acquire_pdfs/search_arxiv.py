

import msgspec
from msgspec.json import decode

class ArXivEntry(msgspec.Struct):
    id: str
    #submitter: str
    authors: str
    title: str
    #comments: str
    #journal_ref: str = msgspec.field(name="journal-ref")
    #report_no: str = msgspec.field(name="report-no")
    categories: str
    abstract: str
    #version: list
    #update_date: str
    #authors_parsed: str
    #license: str | None = None
    doi: str | None = None
    
decoder = msgspec.json.Decoder(type=ArXivEntry)
with open("../arxiv/arxiv-metadata-oai-snapshot.json", "rb") as f:
    data = decoder.decode_lines(f.read())


#encoder = msgspec.json.Encoder()
#with open("../arxiv/sample2.json", "wb") as f:
#    f.write(encoder.encode(data[:1000]))
