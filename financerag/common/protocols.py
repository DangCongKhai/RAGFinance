from abc import ABC, abstractmethod
from typing import List, Dict

__all__=[
    "Retrieval"
]

class Retrieval(ABC):
    @abstractmethod
    def retrieve(self,
            queries : Dict[str, str],             
            top_k : int,
    ):
        """An abstract method that is used to retrieve the top_k-most relevant corpus to the given query.
        The returned argument is retrieved results which is a dictionary that has query as key and a dictionary of 
        corpus with its relevance score as value

        Args:
            queries (Dict[str, str]): A dictionary of query
            top_k (int): The number of documents to retrieve

        Returns:
            retrieved_results (Dict[str, Dict[str, float]])
        """
        pass

    