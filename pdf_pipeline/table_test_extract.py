from gmft import CroppedTable, TableDetector, AutoTableFormatter
from gmft.pdf_bindings import PyPDFium2Document

detector = TableDetector()
formatter = AutoTableFormatter()

pdf_path = '/home/louis/data/raw_data/pdf/10.1103/physrevb.10.4572.pdf'

def ingest_pdf(pdf_path): # produces list[CroppedTable]
    doc = PyPDFium2Document(pdf_path)
    tables = []
    for page in doc:
        tables += detector.extract(page)
    return tables, doc

tables, doc = ingest_pdf(pdf_path)
doc.close() # once you're done with the document


