from tqdm import tqdm 
from typing import Dict, List, Tuple
from ..common import Retrieval
import re
from joblib import Parallel, delayed
import numpy as np
import os
from nltk.tokenize import word_tokenize
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.documents import Document
PATTERN = r"([a-zA-Z]+|\d{4})"

def tokenize_regex(word):
    return re.findall(PATTERN, word)

class BM25_Retriever(Retrieval):
    def __init__(self,
    ):
        pass
    def retrieve(self,
            queries : Dict[str, str],   
            corpus : Dict[str, str],         
            top_k : int = 10, 
    ):
        documents = self._create_document_from_corpus(corpus)
        self.retriever = BM25Retriever.from_documents(documents = documents, k = top_k, preprocess_func=word_tokenize)
        retrieved_results = {}
        
        for query_id, query in queries.items():
            retrieved_docs = self.retriever.invoke(query)
            doc_dict = {doc.id : doc.page_content for doc in retrieved_docs}
            retrieved_results[query_id] = doc_dict

        return retrieved_results        



    def _create_document_from_corpus(self, corpus : Dict[str, str]):
        # Preprocess doc first
        corpus_processed = {corpus_id : ' '.join(re.findall(PATTERN, text)) for corpus_id, text in corpus.items()}
        documents = [Document(page_content = text, id = corpus_id) for corpus_id, text in corpus_processed.items()]
        return documents


