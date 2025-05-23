import os
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from ..common import Retrieval
from typing import Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm

class DenseRetrieval(Retrieval):
    def __init__(self, vector_store : VectorStoreRetriever, dataset_name : str):
        """Initialize the dense retrieval

        Args:
            vector_store (VectorStoreRetriever): any favorite vector store retrieval of langchain
            dataset_name (str) : dataset name of the corpus you want to retrieve the result
        """
        self.vector_store = vector_store
        self.dataset_name = dataset_name
    
    def retrieve(self,
            queries : Dict[str, str],    
            top_k : int = 10,
    ):  
        """This function is used to retrieve the top_k most relevant text to each of the query in the query dictionary.k

        Args:
            queries (Dict[str, str]): A dictionary of query
            top_k (int, optional): The number of results to retrieve for each query. Defaults to 10.

        Returns:
            retrieved_results (Dict[str, Dict[str, float]]) : A dictionary contains the query_id as key with a dictionary of corpus_id and its score as value
        """
        assert len(self.vector_store.similarity_search(query = "Hello", k = 1)) != 0, "You must load corpus to your Retrieval first using: load_corpus_for_searching(corpus)"

        retrieved_results = {}
        for query_id, query in tqdm(queries.items(), desc = "Retrieving result:"):
            retrieved_docs_with_score = self.vector_store.similarity_search_with_score(query = query, k = top_k, filter = {'dataset_name' : self.dataset_name})
            docs_dict = {}
            for doc, score in retrieved_docs_with_score:
                docs_dict[doc.id] = score.item()
            retrieved_results[query_id] = docs_dict
        
        return retrieved_results
        

    def load_corpus_for_searching(self, 
                                corpus : Dict[str, str],
                                saved_index : bool = False):
        """This function is used to load corpus into the vector store of dense retrieval so that it can retrieve
        documents from a given query

        Args:
            corpus (Dict[str, str]): A dictionary of corpus
            index_saved_path (str, optional): Path where you want to save or load existing index if you are using FAISS as vector store. Defaults to None.
            saved_index (bool, optional): Specify if you want to save your index if you are using FAISS vector store. Defaults to False.
        """

        # What about other
        saved_path = f'faiss_index/{self.dataset_name}_index'
        if os.path.exists("faiss_index"):
            if os.path.exists(saved_path) and type(self.vector_store).__name__ == 'FAISS':
                self.vector_store = FAISS.load_local(
                    saved_path, GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004"), allow_dangerous_deserialization=True
                )
                return
        else:
            if saved_index:
                os.mkdir('faiss_index')
        corpus_documents = []
        for id, text in tqdm(corpus.items(), desc='Loading document'):
            corpus_documents.append(Document(page_content = text, id = id, metadata = {'dataset_name' :  self.dataset_name}))
        self.vector_store.add_documents(corpus_documents)

        if saved_index and type(self.vector_store).__name__ == 'FAISS':
            self.vector_store.save_local(saved_path)
            print(f"Successfully saved index of vector store to path : {saved_path}")
                
        


