import logging 
from time import time
from functools import wraps
import pandas as pd
from typing import Dict, List, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename='process.log')
formatter = logging.Formatter("%(asctime)s :%(message)s", datefmt = '%d/%m/%Y %I:%M:%S %p')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

__all__ = [
    'timer', 
    'get_query_and_retrieved_corpus_text',
    'process_retrieval_df',
    'get_final_result'
]

def timer(func):
    @wraps(func)
    def measure_time(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        total_time_taken = t2 - t1
        #  Things to log: Loading docoment time + Retrieval time + Reranking time
        if 'retrieve' in func.__name__:
            top_k = kwargs.get('top_k', None)
            queries = kwargs['queries']
            query_len = len(queries)
            if top_k is None:
                top_k = args[-1]
            avg_time = total_time_taken / query_len
            logger.info(f"Retrieval time for {query_len} queries, {top_k} docs/query = {round(avg_time, 4)}s")
        elif 'load' in func.__name__:
            corpus = kwargs.get('corpus', None)
            if corpus is not None:
                corpus_len = len(corpus)
                avg_time = total_time_taken / corpus_len
                logger.info(f"Loading {corpus_len} documents - Time taken for 1 doc = {avg_time}s")
            else:
                logger.info(f"Loading documents = {total_time_taken}s")
        elif 'rerank' in func.__name__:
            retrieved_result: Dict[str, Dict[str, float]] = kwargs.get('retrieved_result', None)
            if retrieved_result is None:
                retrieved_result = args[0]
            if isinstance(retrieved_result, Dict):
                query_len = len(retrieved_result)
                doc_len = len(list(retrieved_result.values())[0])
                avg_time = total_time_taken / query_len
                logger.info(f"Reranking for {query_len} queries - Time taken for reranking {doc_len} docs/query = {avg_time}s")
           
        return result
    return measure_time




def get_query_and_retrieved_corpus_text(queries : Dict[str, str], corpus : Dict[str, str], retrieved_result : Dict[str, Dict[str, float]], n_query : int = 2):
    """Print out query and their retrieved corpus

    Args:
        queries (Dict[str, str]): _description_
        corpus (Dict[str, str]): _description_
        retrieved_result (Dict[str, Dict[str, float]]): _description_
        n_query (int, optional): _description_. Defaults to 2.
    """
    for i, (query_id, docs) in enumerate(retrieved_result.items()):
        if (i == n_query): break
        query = queries[query_id]
        print(f"Query : {query}")
        print("Retrieve documents: ")
        for rank, corpus_id in enumerate(docs):
            doc = corpus[corpus_id]
            print(f"Top {rank+1}: \n {doc}")
        print('*' * 100)


    
def load_query(dataset_name : str, new_path_to_query = None) -> Dict[str, str]:
    query_df = pd.read_json(f"../../finance_dataset/{dataset_name.lower()}_queries.jsonl/queries.jsonl", lines = True)
    # Convert into Dict[str, str]
    query_dict = {row['_id'] : row['text'] for i, row in query_df.iterrows()}
    return query_dict

    
def load_corpus(dataset_name : str, new_path_to_corpus = None) -> Dict[str, str]:
    corpus_df = pd.read_json(f"finance_dataset/{dataset_name.lower()}_corpus.jsonl/corpus.jsonl", lines = True)
    # Convert into Dict[str, str]
    corpus_dict = {row['_id'] : row['text'] for i, row in corpus_df.iterrows()}
    return corpus_dict

def process_retrieval_df(result : pd.DataFrame) -> Dict[str, Dict[str, float]]:
    retrieved_result : Dict[str, Dict[str, float]] = {}
    for i, row in result.iterrows():
        query_id, corpus_id = row['query_id'], row['corpus_id']
        if query_id not in retrieved_result:
            retrieved_result[query_id] = {}
        retrieved_result[query_id][corpus_id] = 1.0
    return retrieved_result


def get_final_result(dataset_names : List[str], method_name : str):
    """This function is used to collect final results for all task with a given method

    Args:
        task_names (List[str]): a list containing all task names
        method_name (str): method for the result you are retrieving

    Returns:
        final_result [pd.DataFrame] : a dataframe containing retrieved results for all tasks
    """
    final_result = pd.DataFrame(columns = ['query_id', 'corpus_id'])
    for dataset_name in dataset_names:
        df = pd.read_csv(f"financerag_result/{method_name}/{dataset_name.lower()}_result.csv")
        final_result = pd.concat([final_result, df], axis = 0)

    return final_result





