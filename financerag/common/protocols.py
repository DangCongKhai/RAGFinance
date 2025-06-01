from abc import ABC, abstractmethod
from typing import List, Dict
from pydantic import validate_call
__all__=[
    "Retrieval",
    "Reranker"
]

class Retrieval(ABC):

    @validate_call
    @abstractmethod
    def retrieve(self,
                queries : Dict[str, str],             
                top_k : int,
                *args, **kwargs
    )-> Dict[str, Dict[str, float]] : 
        """An abstract method that is used to retrieve the top_k-most relevant corpus to the given query.
        The returned argument is retrieved results which is a dictionary that has query as key and a dictionary of 
        corpus with its relevance score as value

        Args:
            queries (Dict[str, str]): A dictionary of query
            top_k (int): The number of documents to retrieve

        Returns:
            (Dict[str, Dict[str, float]]): A dictionary where each query ID maps to a dictionary containing the top K most relevant corpus IDS and their corresponding relevance scores, sorted in descending order                                                    
        """
        pass


class Reranker(ABC):

    @abstractmethod
    def __init__(self, queries : Dict[str, str], corpus : Dict[str, str], *args, **kwargs):
        pass
    
    @validate_call
    @abstractmethod
    def rerank(self, 
               retrieved_result : Dict[str, Dict[str, float]], 
               top_k : int = 10,
               batch_size : int = 32, *args, **kwargs) -> Dict[str, Dict[str, float]]:
        """Rerank the retrieved documents based on their relevance scores (semantic meaning)

        Args:
            retrieved_result (Dict[str, Dict[str, float]]): Result that you retrieved from your DenseRetrieval or BM25 or both(after RAG Fusion)
            top_k (int, optional): The number of documents to retrieve after reranking. Defaults to 10.
            batch_size (int, optional): The number of documents to load during reranking
        Returns:
            (Dict[str, Dict[str, float]]) : A dictionary where we get the top-k most relevant documents with their relevance scores for each query
        """
        pass
    