import pandas as pd
from typing_extensions import List


class FinanceDataLoader:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load(self, corpus_path : str = None, query_path : str = None):
        """This function is used to load queries and corpus from the corresponding

        Args:
            corpus_path (str, optional): Root path to corpus. Defaults to None.
            query_path (str, optional): Root path to query. Defaults to None.
        Returns:
            queries (Dict[str, str]) : Dataset queries 
            corpus : Dict[str, str] : Dataset corpus
        """
        corpus, queries = self.load_corpus(corpus_path), self.load_queries(query_path)
        return queries, corpus
    
    def load_queries(self, query_path : str = None):
        """This function is used to load corpus data from the given dataset name

        Args:
            query_path (str, optional): Root path to query. Defaults to None.

        Returns:
            queries (Dict[str, str]) : Queries dictionary
        """
        if query_path is None:
            query_path = 'finance_dataset'
        queries_df = pd.read_json(f"{query_path}/{self.dataset_name}_queries.jsonl/queries.jsonl", lines = True)
        queries_df['text_concat'] = queries_df['text']
        if 'title' in queries_df.columns:
            queries_df['text_concat'] = queries_df['title'] + ' ' + queries_df['text']
            queries_df['text_concat'] = queries_df['text_concat'].str.strip()
        queries = {row['_id'] : row['text_concat'] for _, row in queries_df.iterrows()}
        return queries

    def load_corpus(self, corpus_path : str = None):
        """This function is used to load corpus data from the given dataset name

        Args:
            corpus_path (str, optional): Root path to corpus. Defaults to None.
            
        Returns:
            corpus (Dict[str,str]) : Corpus dictionary
        """
        if corpus_path is None:
            corpus_path = 'finance_dataset'
        corpus_df = pd.read_json(f'{corpus_path}/{self.dataset_name}_corpus.jsonl/corpus.jsonl', lines = True)
        # Filter rows with the same id and without text
        corpus_df.drop_duplicates(inplace = True)
        corpus_df['text_concat'] = corpus_df['text']
        if 'title' in corpus_df.columns:
            corpus_df['text_concat'] = corpus_df['title'] + ' ' + corpus_df['text']
            corpus_df['text_concat'] = corpus_df['text_concat'].str.strip()
        corpus_df['text_len'] = corpus_df['text_concat'].str.len()
        # Extract the row with text greater than 0
        corpus_df = corpus_df.loc[corpus_df['text_len'] > 0]
        corpus = {row['_id'] : row['text_concat'] for _, row in corpus_df.iterrows()}
        return corpus


