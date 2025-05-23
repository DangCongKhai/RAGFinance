from typing import List, Dict, Literal
from .TaskMetadata import TaskMetaData
from ..common import FinanceDataLoader, Retrieval
import pandas as pd
import os
class BaseTask:
    def __init__(self, metadata : TaskMetaData):
        self.metadata = metadata
        self.queries : Dict[str, str] = None
        self.corpus : Dict[str, str] = None
        self.retrieved_results : Dict[str, Dict[str, float]] = None

    def load(self):
        """This function is used to load the query and corpus of the corresponding dataset name
        """
        dataset_name = self.metadata.dataset_name
        finance_loader = FinanceDataLoader(dataset_name = dataset_name)
        self.queries, self.corpus = finance_loader.load()

    def retrieve(self,
                 retriever : Retrieval,
                 top_k : int = 10) -> Dict[str, Dict[str, float]] :
        """This function is used to retrieve the top_k most relevant corpus id with its score for each query that we have loaded for each dataset

        Args:
            retriever (Retrieval): Retriever for your task. It could be a DenseRetrieval or SparseRetrieval
            top_k (int, optional): The number of k most relevant corpus to each query to be retrieved. Defaults to 10.

        Returns:
            retrieved_resulst (Dict[str, Dict[str, float]]) : A dictionary contains query_id with its top-k most relevant corpus id and the corresponding score
        """
        assert self.queries is not None and self.corpus is not None, "Corpus and query is missing! You must call 'load' function first before retrieving result for your quuery"
        self.retrieved_results = retriever.retrieve(queries = self.queries, top_k = top_k)
        return self.retrieved_results
    

    def save_retrieved_results(self):
        """This function is used to save your retrieved results to csv file for final submission!
        """
        assert self.retrieved_results is not None, "Your retrieved results are not available! You must load your corpus first and then retrieve the result to use this function!"
        final_result_dict = {
            'query_id' : [],
            'corpus_id' : []
        }
        for query_id, corpus_dict in self.retrieved_results.items():
            corpus_ids = list(corpus_dict.keys())
            final_result_dict['query_id'].extend([query_id] * len(corpus_ids))
            final_result_dict['corpus_id'].extend(corpus_ids)
        
        final_result_df = pd.DataFrame(final_result_dict)

        if not os.path.isdir("financerag_result"):
            os.mkdir('financerag_result')
        
        save_path = f"./financerag_result/{self.metadata.dataset_name}_result.csv"
        final_result_df.to_csv(save_path, index = False)
        print(f"Saved result successfully to {save_path}!")

