from tqdm import tqdm
from financerag.common.helper import timer
from .knowledge_graph import KG
from .utils import create_vector_store

import numpy as np
from typing_extensions import Dict, List, Tuple
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

class HippoRAGV2:
    def __init__(
        self,
        triples_vector_store: VectorStore = None,
        passages_vector_store: VectorStore = None,
        is_directed : bool = True
    ):

        if triples_vector_store is None:
            triples_vector_store = create_vector_store()
        self.triples_vector_store = triples_vector_store
        if passages_vector_store is None:
            passages_vector_store = create_vector_store()
        self.passages_vector_store = passages_vector_store
        self.KG = KG(is_directed=is_directed)

    @timer
    def offline_indexing(self, corpus: Dict[str, str],add_synonymy_edge = False, batch_size=4):

        corpus_id = corpus.keys()
        corpus_text = corpus.values()

        for i in tqdm(range(0, len(corpus_id), batch_size), desc="Ascending"):
            batch_corpus_id = corpus_id[i : i + batch_size]
            batch_corpus_text = corpus_text[i : i + batch_size]

            # Passages handling
            self.passages_vector_store.add_texts(
                batch_corpus_text,
                metadatas=[{"id": id_} for id_ in batch_corpus_id],
            )
            self.KG.add_vertices(batch_corpus_id)

            batch_triples = self._extract_triples(batch_corpus_text)

            for batch_idx, triples in enumerate(batch_triples):
                
                triples_text = []
                entities = set()
                for triple in triples:
                    triples_text.append(" ".join(triple))
                    entities.add(triple[0])
                    entities.add(triple[2])

                self.triples_vector_store.add_texts(
                    triples_text, metadatas=[{"triple": triple} for triple in triples]
                )

                self.KG.add_vertices(entities)
                # Add edges between entities
                self.KG.add_edges(triples)

                # Add edges between those entities with passage
                passage_entities_edges = [
                    [entity, "contained", batch_corpus_id[batch_idx]]
                    for entity in entities
                ]
                self.KG.add_edges(passage_entities_edges)
                
            if add_synonymy_edge:
                # Handling adding synonymy edge here
                pass
            
    def _add_synonymy_edges(self):
        pass        

    def _extract_triples(self, corpus_text: List[str]):
        pass

    def retrieve(self, query : Dict[str, str], top_k = 10):
        
        # Handling query in batch
        retrieved_result = {}
        
        for query_id, query in tqdm(query.items(), desc="Processing query"):
            
            # Get top_k most related passages
            retrieved_docs_with_score = self.passages_vector_store.similarity_search_with_score(query, k = top_k)
            docs_dict = {}
            for doc, score in retrieved_docs_with_score:
                if doc.metadata['id'] not in docs_dict:
                    docs_dict[doc.metadata['id']] = score.item()
            
            # Get query_to_triples, how many triples do we retrieved
            retrieved_triples : Tuple[Document, float] = self.triples_vector_store.similarity_search_with_score(query, k = 5)
            
            filtered_triples = self._filter_triples_by_llm(query, retrieved_triples)
            
            if not filtered_triples: # No triples are left
                retrieved_result[query_id] = docs_dict
            else:
                phrase_nodes_weights = np.zeros(shape = (self.KG.get_number_of_vertices()))
                passage_nodes_weights = np.zeros(shape = (self.KG.get_number_of_vertices()))
                
                # Work on phrase nodes weight frist
                
                # 
                
                
                # Get the entities within the filtered_triples
                phrase_nodes = self._get_phrase_nodes(filtered_triples)
                # https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/hipporag/HippoRAG.py#L1502
                # Score of query node = avg of similarity score of triples containing it
                
                
                
                # Assign probability mass for query node
                
                # Get similarity score of 
                
                # Get nodes from filtered triples
                
                # Get passage nodes from retrieved passage
                
                # Assign probability or reset vector
                
                # Run personalized page rank
                
                # Get rank of top_k passages in descending order
                pass
    def _get_phrase_node(filtered_triples):
        phrase_nodes = set()
        for tuple in filtered_triples:
            document = tuple[0]
            triple = document.metadata['triple']
            phrase_nodes.add(triple[0])
            phrase_nodes.add(triple[1])
        return phrase_nodes
            
    def _filter_triples_by_llm(query, retrieved_triples):
        
        # Create a chain: prompt | llm | output parser -> Tuple[Document, float]
        
        # Temporarily returned the same retrieve triples
        return retrieved_triples