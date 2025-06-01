"""
What to do this morning
- Run our Dense Retrieval agin to see if result if improved!
- Create a reranker class
- Questions:
    1. Do we need to create a base class for reranker? If yes, what are the abstract methods that we should implement?
    2. Consider add RAG FUSION -> In the end, the reranker only receives final result [query_id : {corpus_id : score}] and rerank based on their similarity 
    with the original query.
    3. General process for implementing reranker
        - Takes in the original query and corpus Dict[str, str] and the retrieved result that we have had + Reranker model
        - Choose a sentence transformer
        - For each query, construct into the format: query, List[text]:
            - Get the similarity score of that query with all retrieved documents 
            - Extract top_k element following the final result
        -> Consider doing parallelization because you are running on you own GPU
    4. Watch a tutorial on when you should we use: Multiprocecssing, Pool, Threading or Parallel from joblib -> Ask ChatGPT first to get general idea

"""

from ..common import Reranker, timer
from sentence_transformers import CrossEncoder
from typing import Dict, List
from pydantic import validate_call

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
        for query_id, doc_dict in retrieved_result.items():
            query = self.queries[query_id]
            idx_to_corpus_dict = {idx : corpus_id for idx, corpus_id in enumerate(doc_dict.keys())}
            documents = [self.corpus[corpus_id] for corpus_id in doc_dict.keys()]
            
            ranks = self.reranker.rank(query = query, documents = documents, return_documents = True, top_k = top_k)
            top_k_docs = {idx_to_corpus_dict[corpus['corpus_id']] : corpus['text'] for corpus in ranks }
            
            # Save results
            final_result[query_id] = top_k_docs
        return final_result        
