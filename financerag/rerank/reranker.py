from ..common import Reranker, timer
from sentence_transformers import CrossEncoder
from typing import Dict, List
from pydantic import validate_call
from tqdm import tqdm

class CrossEncoderReranker(Reranker):
    def __init__(self, 
                 queries : Dict[str, str], 
                 corpus : Dict[str, str], 
                 reranker : CrossEncoder, 
                 batch_size : int = 32, 
                 *args, **kwargs):
        self.queries = queries
        self.corpus = corpus
        self.reranker = reranker
        self.batch_size = batch_size

    @timer
    def rerank(self, 
               retrieved_result : Dict[str, Dict[str, float]], 
               top_k : int = 10,) -> Dict[str, Dict[str, float]] :
        
        final_result = {}
        for query_id, doc_dict in tqdm(retrieved_result.items(), desc = 'Reranking'):
            query = self.queries[query_id]
            idx_to_corpus_dict = {idx : corpus_id for idx, corpus_id in enumerate(doc_dict.keys())}
            documents = [self.corpus[corpus_id] for corpus_id in doc_dict.keys()]
            
            ranks = self.reranker.rank(query = query, documents = documents, return_documents = True, top_k = top_k)
            top_k_docs = {idx_to_corpus_dict[corpus['corpus_id']] : corpus['score'] for corpus in ranks }
            
            # Save results
            final_result[query_id] = top_k_docs
        return final_result        
