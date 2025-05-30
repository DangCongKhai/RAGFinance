import os
from ..common import Retrieval
from typing import Dict, List, Tuple
from tqdm import tqdm
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters.base import TextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_community.vectorstores import FAISS


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
            **kwargs
    ):  
        """This function is used to retrieve the top_k most relevant text to each of the query in the query dictionary.k

        Args:
            queries (Dict[str, str]): A dictionary of query
            top_k (int, optional): The number of results to retrieve for each query. Defaults to 10.

        Returns:
            retrieved_results (Dict[str, Dict[str, float]]) : A dictionary contains the query_id as key with a dictionary of corpus_id and its score as value
        """
        assert len(self.vector_store.similarity_search(query = "Hello", k = 1)) != 0, "You must load corpus to your Retrieval first using: load_corpus_for_searching(corpus)"
        
        top_k = max(10, top_k)
        retrieved_results = {}
        for query_id, query in tqdm(queries.items(), desc = "Retrieving result:"):
            retrieved_docs_with_score = self.vector_store.similarity_search_with_score(query = query, k = top_k, filter = {'dataset_name' : self.dataset_name})
            docs_dict = {}
            for doc, score in retrieved_docs_with_score:
                if (len(docs_dict.keys()) == 10): 
                    break
                if doc.metadata['id'] not in docs_dict:
                    docs_dict[doc.metadata['id']] = score.item()
            # Here are things to consider:
            # 1. Averaging all top_k document score and sort it according to that score
            # 2. Rerank document
            print(f"Len key of docs dict :{len(docs_dict)}")
            retrieved_results[query_id] = docs_dict
        
        return retrieved_results
        

    def load_corpus_for_searching_without_splitting(self, 
                                corpus : Dict[str, str],
                                saved_index : bool = False):
        """This function is used to load entire corpus text into the vector store of dense retrieval so that it can retrieve
        documents from a given query

        Args:
            corpus (Dict[str, str]): A dictionary of corpus
            index_saved_path (str, optional): Path where you want to save or load existing index if you are using FAISS as vector store. Defaults to None.
            saved_index (bool, optional): Specify if you want to save your index if you are using FAISS vector store. Defaults to False.
        """
        

        corpus_documents = []
        for id, text in tqdm(corpus.items(), desc='Loading document'):
            corpus_documents.append(Document(page_content = text, metadata = {'dataset_name' :  self.dataset_name, 'id' : id}))
        self.vector_store.add_documents(corpus_documents)
        self._save_index(saved_index)
        

    def load_corpus_from_existing_index(self, index_path : str):
        """This function is used to load existing index 

        Args:
            index_path (str): Path to your preloaded index
        """
        assert os.path.exists(index_path), f"{index_path} does not exist"
        if type(self.vector_store).__name__ == 'FAISS':
            self.vector_store = FAISS.load_local(
                index_path, GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004"), allow_dangerous_deserialization=True
            )    
        else:
            print(f"You must use FAISS as vector database in order ot use this function!")
    
    
    def load_corpus_with_splitting(self,
                                   text_splitter : TextSplitter,
                                   corpus : Dict[str, str],
                                   saved_index = False):
        documents = []
        for corpus_id, text in tqdm(corpus.items(), desc = 'Loading document:'):
            # Split documents
            texts = text_splitter.split_text(text)
            split_documents = [Document(page_content = split_text, metadata = {'dataset_name' :  self.dataset_name, 'id' : corpus_id}) for split_text in texts]
            documents.extend(split_documents)
        self.vector_store.add_documents(documents)
        self._save_index(saved_index)


    def _save_index(self, saved_index : bool):
        """Helper function that is used to save index in case using FAISS Index

        Args:
            saved_index (bool): Bool value that specifies whether you wants to save your faiss index
        """
        saved_path = f'faiss_index/{self.dataset_name}_index'
        if saved_index and type(self.vector_store).__name__ == 'FAISS':
            if not os.path.exists('faiss_index'):
                os.mkdir('faiss_index')
            self.vector_store.save_local(saved_path)
            print(f"Successfully saved index of vector store to path : {saved_path}")




