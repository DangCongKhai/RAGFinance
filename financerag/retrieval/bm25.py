from tqdm import tqdm 
from typing import Dict, List, Tuple
from ..common import Retrieval
import re
from joblib import Parallel, delayed
import numpy as np
import os
from nltk.tokenize import word_tokenize

PATTERN = r"([a-zA-Z]+|\d{4})"

class BM25(Retrieval):
    def __init__(self,
                 k : int = 1.2,
                 b : int = 0.75
    ):
        self.k = k
        self.b = b
        self.corpus_keywords = None
        self.frequency_arr = None
        self.vocab_to_idx = None
        self.avg_doc_len = None
    def retrieve(self,
            queries : Dict[str, str],   
            corpus : str,          
            top_k : int = 10, 
    ):
        # Consider doing the query in parallel as we don't need to call any API
        corpus_keywords = {id : re.findall(PATTERN, text.lower()) for id, text in corpus.items()}
        self._create_frequency_arr(corpus_keywords)
        self.avg_doc_len = self._get_avg_document_length()
        
        retrieved_result = {}
        corpus_retrieved_result : List[Dict[str, float]] = Parallel(n_jobs = os.cpu_count() - 4)(delayed(self._retrieve_corpus_for_query)(query, corpus_keywords, top_k) for query in queries.values())
        # for query_id, query in queries.items():
        #     corpus_retrieved = self._retrieve_corpus_for_query(query, corpus_keywords, top_k)
        #     retrieved_result[query_id] = corpus_retrieved
        for query_id, corpus_retrieved in zip(queries.keys(), corpus_retrieved_result):
            retrieved_result[query_id] = corpus_retrieved

        return retrieved_result
        

    def _create_frequency_arr(self, corpus_keywords : Dict[str, List[str]]):
        # Get vocabulary 
        vocab = set()
        for keywords in corpus_keywords.values():
            vocab = vocab.union(set(keywords))

        # Construct vocab_to_index dictionary
        self.vocab_to_idx = {word : idx for idx, word in enumerate(vocab)}
    
        # Construct zeros frequency arr with shape (n_corpus, vocab_size)
        n_corpus, vocab_size = len(corpus_keywords), len(vocab)
        self.frequency_arr = np.zeros(shape = (n_corpus, vocab_size))
        
        # Calculate the frequency of keywords in our corpus
        for i, keywords in enumerate(tqdm(corpus_keywords.values(), desc = 'Create frequency_arr')):
            for keyword in keywords:
                idx = self.vocab_to_idx[keyword]
                self.frequency_arr[i, idx] += 1
        
    def _get_avg_document_length(self):
        """Get average length of the corpus

        Returns:
            (float): Average length of the corpus
        """
        return np.mean(np.sum(self.frequency_arr, axis = 1)).item()


    def _IDF(self, word : str):
        """Calculate Inverse Document Frequency of a word

        Args:
            word (str): word for calculating IDF

        Returns:
            idf (float): IDF score for that word
        """
        idx = self.vocab_to_idx.get(word, None)
        if idx is None: 
            return 0
        n_documents, n_documents_containing_word = len(self.frequency_arr), self.frequency_arr[:, idx].sum()
        idf = np.log((n_documents - n_documents_containing_word + 0.5) / (n_documents_containing_word + 0.5) + 1).item()
        return idf
        
    
    def _TF(self, word: str, corpus_idx : int):
        idx = self.vocab_to_idx.get(word, None)
        if idx is None:
            return 0
        frequency = self.frequency_arr[corpus_idx, idx]
        return frequency

    
    def _bm25_score(self, query : str, corpus_idx : str, corpus_text : List[str]):
        # Calculate BM25 score of the query with all corpus in sorted order
        words = query.lower().split(' ') # Consider using a more advanced splitter here
        total_score = 0
        for word in words:
            tf, idf = self._TF(word, corpus_idx), self._IDF(word)
            total_score += idf * tf * (self.k + 1) / (tf + self.k * (1 - self.b + self.b * len(corpus_text) / self.avg_doc_len))
        return total_score.item()

    def _retrieve_corpus_for_query(self, query : str, corpus_keywords : Dict[str, List[str]], top_k :int):
        # Calculate the bm25 score for the query with all docs
        # Extract only top_k documents!
        result = {}
        for corpus_idx, (corpus_id, corpus_text) in enumerate(corpus_keywords.items()):
            similarity_score = self._bm25_score(query, corpus_idx = corpus_idx, corpus_text=corpus_text)
            result[corpus_id] = similarity_score
        
        # Sort the result
        sorted_result = sorted(result.items(), key = lambda item : item[1])
        top_k_sorted_result = sorted_result[:top_k]
        corpus_dict = {key : value for key, value in top_k_sorted_result}
        return corpus_dict
    
    # Write another BM25 using Bm25 retriever instead
    



