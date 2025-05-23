import pandas as pd
from typing_extensions import List


class FinanceDataLoader:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load(self):
        """This function is used to load queries and corpus from the corresponding

        Returns:
            queries (Dict[str, str]) : Dataset queries 
            corpus : Dict[str, str] : Dataset corpus
        """
        corpus, queries = self.load_corpus(), self.load_queries()
        return queries, corpus
    
    def load_queries(self):
        """This function is used to load corpus data from the given dataset name

        Returns:
            queries (Dict[str, str]) : Queries dictionary
        """
        queries_df = pd.read_json(f"finance_dataset/{self.dataset_name}_queries.jsonl/queries.jsonl", lines = True)
        queries = {row['_id'] : row['text'] for _, row in queries_df.iterrows()}
        return queries

    def load_corpus(self):
        """This function is used to load corpus data from the given dataset name

        Returns:
            corpus (Dict[str,str]) : Corpus dictionary
        """
        corpus_df = pd.read_json(f'finance_dataset/{self.dataset_name}_corpus.jsonl/corpus.jsonl', lines = True)
        # Filter rows with the same id and without text
        corpus_df.drop_duplicates(inplace = True)
        corpus_df['text_len'] = corpus_df['text'].apply(lambda x : len(x))
        # Extract the row with text greater than 0
        corpus_df = corpus_df.loc[corpus_df['text_len'] > 0]
        corpus = {row['_id'] : row['text'] for _, row in corpus_df.iterrows()}
        return corpus


