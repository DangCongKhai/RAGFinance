from typing import List, Dict, Literal
from .TaskMetadata import TaskMetaData
from ..common import FinanceDataLoader, Retrieval, timer
import pandas as pd
import os
from pydantic import validate_call
class BaseTask:
    def __init__(self, metadata : TaskMetaData):
        self.metadata = metadata
        self.queries : Dict[str, str] = None
        self.corpus : Dict[str, str] = None
        self.retrieved_result : Dict[str, Dict[str, float]] = None

    def load(self):
        """This function is used to load the query and corpus of the corresponding dataset name
        """
        dataset_name = self.metadata.dataset_name
        finance_loader = FinanceDataLoader(dataset_name = dataset_name)
        self.queries, self.corpus = finance_loader.load()


    def retrieve(self,
                 retriever : Retrieval,
                 top_k : int = 10,
                 *args, **kwargs) -> Dict[str, Dict[str, float]] :
        """This function is used to retrieve the top_k most relevant corpus id with its score for each query that we have loaded for each dataset

        Args:
            retriever (Retrieval): Retriever for your task. It could be a DenseRetrieval or SparseRetrieval
            top_k (int, optional): The number of k most relevant corpus to each query to be retrieved. Defaults to 10.

        Returns:
            retrieved_result (Dict[str, Dict[str, float]]) : A dictionary contains query_id with its top-k most relevant corpus id and the corresponding score
        """
        assert self.queries is not None and self.corpus is not None, "Corpus and query is missing! You must call 'load' function first before retrieving result for your quuery"
        self.retrieved_result = retriever.retrieve(queries = self.queries, corpus = self.corpus, top_k = top_k)
        return self.retrieved_result
    
    @validate_call
    def save_retrieved_results(self, retrieved_result : Dict[str, Dict[str, float]]):
        
        """This function is used to save your retrieved results to csv file for final submission!

        Arguments: 
            retrieved_result (Dict[str, Dict[str, float]]): a dictionary where each query id is map to the top-k most relevant documents with their relevance score!
        """
        
        final_result_dict = {
            'query_id' : [],
            'corpus_id' : []
        }
        for query_id, corpus_dict in retrieved_result.items():
            corpus_ids = list(corpus_dict.keys())
            final_result_dict['query_id'].extend([query_id] * len(corpus_ids))
            final_result_dict['corpus_id'].extend(corpus_ids)
        
        final_result_df = pd.DataFrame(final_result_dict)

        if not os.path.isdir("financerag_result"):
            os.mkdir('financerag_result')
        
        save_path = f"./financerag_result/{self.metadata.dataset_name}_result.csv"
        final_result_df.to_csv(save_path, index = False)
        print(f"Saved result successfully to {save_path}!")

