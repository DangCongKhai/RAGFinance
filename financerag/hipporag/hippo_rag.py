import numpy as np
from typing import Dict, List, Union
from tqdm import tqdm
import logging

from ..common import Retrieval, timer
from .utils import PROMPT_TEMPLATE, CustomizedListParser
from .knowledge_graph import KG


from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from langchain_core.vectorstores import VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv("../../.env")
LLM = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename='process.log')
formatter = logging.Formatter("%(asctime)s :%(message)s", datefmt = '%d/%m/%Y %I:%M:%S %p')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class HippoRAG(Retrieval):
    def __init__(self, retriever : VectorStore, model_id : str = "dslim/bert-base-NER"):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id)
        self.ner_extractor = pipeline(task = 'ner', model = model, tokenizer = tokenizer, aggregation_strategy = 'max')
        self.relations_extractor = PROMPT_TEMPLATE | LLM | CustomizedListParser()
        self.KG = KG()
        self.retriever = retriever

    @timer
    def retrieve(self, queries : Dict[str, str], corpus : Dict[str, str], top_k = 10, *args, **kwargs) -> Dict[str, Dict[str, float]]:
        assert self.KG.get_number_of_vertices() != 0, "You must perform offline indexing first!"
        
        retrieved_result = {}
        for query_id, query in tqdm(queries.items(), desc = 'Retrieving'):
            retrieved_result[query_id] = {}
            # Extract query named entities
            query_named_entities = [entity['word'] for entity in self.ner_extractor(query)]
            if query_named_entities:
                # Determine query node using retriever
                query_nodes = []
                for entity in query_named_entities:
                    query_node = self.retriever.similarity_search(query = entity, k = 1)[0].page_content
                    query_nodes.append(self.KG.get_vertex_idx(query_node))
                
                
                # After having query node, initialize vector n over N nodes
                nodes_probabilities = np.zeros(shape = (self.KG.get_number_of_vertices(),))
                nodes_probabilities[query_nodes] = 1 / len(query_nodes)

                # Multiply by node specificity for each node
                for query_node in query_nodes:
                    nodes_probabilities[query_node] *= 1/self.KG.get_number_of_passages_by_vertex_id(query_node)
                
                # Run personalized page rank
                final_nodes_probabilities = self.KG.run_personalized_page_rank(reset_vector = nodes_probabilities, damping = 0.85)
                # Final rank
                final_corpus_rank = self.matrix_P.T @ final_nodes_probabilities

                # Sort by rank
                top_k_corpus_idx = final_corpus_rank.argsort()[::-1][:top_k]
                corpus_dict = {}
                for idx in top_k_corpus_idx:
                    corpus_id = self.idx_to_corpus_id[idx]
                    score = final_corpus_rank[idx]
                    corpus_dict[corpus_id] = score
                retrieved_result[query_id] = corpus_dict

            else: # No named entity was extracted, implement logic later
                logger.info(f"Cannot extract named entities from query_id = {query_id}")
                # Add random result temporarilily
                for i, corpus_id in enumerate(corpus.keys()):
                    if (i == top_k): break
                    retrieved_result[query_id][corpus_id] = 1.0
        return retrieved_result
    @timer
    def offline_indexing(self, corpus : Dict[str, str], batch_size = 512):
        """Performs offline indexing process to construct KG triples and matrix P used for online retrieval process

        Args:
            corpus (Dict[str, str]): Dictionary of corpus with key is corpus_id and the value is corpus text
            batch_size (int, optional): Batch size to perform indexing. Defaults to 700.
        """
        self.idx_to_corpus_id = {idx : corpus_id for idx, corpus_id in enumerate(corpus.keys())}
        corpus_text = list(corpus.values())
        
        entities_list, relations_list = [], []
        for i in tqdm(range(0, len(corpus_text), batch_size), desc='Loading'):
            batch_corpus = corpus_text[i : i + batch_size]
            batch_entities, batch_relations = self._extract_entities_and_relations(batch_corpus)

            entities_list.extend(batch_entities) 
            relations_list.extend(batch_relations)

        # Add nodes to KG and store number of passages containing that node in each node
        for entities in tqdm(entities_list, desc = 'Add nodes to KG'):
            self.KG.add_vertices(entities)
        
        # Construct a retriever encoder
        nodes = self.KG.get_vertices()
        self.retriever.add_texts(nodes)

        # Construct matrix P with shape |N|x|P|, P[i][j]: number of times node i appears in corpus j
        N, P = self.KG.get_number_of_vertices(), len(corpus_text)
        self.matrix_P = np.zeros(shape = (N, P))
        for corpus_idx, entities in enumerate(tqdm(entities_list, desc = 'Construct matrix P')):
            for entity in entities:
                entity_idx = self.KG.get_vertex_idx(entity)
                occurrence = self.KG.get_number_of_passages_by_vertex_id(entity_idx)
                self.matrix_P[entity_idx, corpus_idx] = occurrence
        
        # Add edges to graph
        for relations in relations_list:
            self.KG.add_edges(relations)
        
       

        
    def _extract_entities_and_relations(self, corpus : List[str]):
        entities_raw_list: List[List[Dict[str, Union[str, float]]]] = self.ner_extractor(corpus)
        message_list = []
        entities_list = []
        for i, entities in enumerate(entities_raw_list):
            one_corpus_text = corpus[i]
            named_entities = [entity['word'] for entity in entities]
            entities_list.append(named_entities)

            one_corpus_message = f"""Paragraph
            ```
            {one_corpus_text}
            ```
            Named_entities = {named_entities}
            
            """
            message_list.append(one_corpus_message)
        message_for_relations_extraction = " ".join(message_list)
        relations_list = self.relations_extractor.invoke({'message' : message_for_relations_extraction})
        return entities_list, relations_list

    def _add_synomymy(self):
        # Add synonymy edges to KG graph
        pass



        