from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining


text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name="nasa-impact/nasa-smd-ibm-st-v2", breakpoint_threshold_type="percentile"))

model = SentenceTransformer("nasa-impact/nasa-smd-ibm-st-v2")


