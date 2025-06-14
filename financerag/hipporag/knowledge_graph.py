import igraph as ig
from igraph import UniqueIdGenerator
from typing import List
import numpy as np

class KG:
    def __init__(self):
        self.graph = ig.Graph(directed = True)
        self.vertex_id_gen = UniqueIdGenerator(initial = list(self.graph.vs))

    def get_number_of_vertices(self):
        return self.graph.vcount()
    def add_vertices(self, vertices : List[str]):
        for vertex in vertices:
            vertex_id = self.vertex_id_gen[vertex]
            if (vertex_id == self.graph.vcount()):
                self.graph.add_vertex(vertex)
                self.graph.vs[vertex_id]['n_passages'] = 0
            self.graph.vs[vertex_id]['n_passages'] += 1

    def add_edges(self, edges : List[List[str]]):
        vertices = set(self.graph.vs['name'])
        for triple in edges:
            if (triple[0] not in vertices or triple[1] not in vertices): continue
            edge_id = self.graph.get_eid(triple[0], triple[2], error = False)
            if edge_id == -1:
                self.graph.add_edge(triple[0], triple[2])

    def run_personalized_page_rank(self, reset_vector = None, damping = 0.85, *args, **kwargs):
        personalized_page_rank = np.array(self.graph.personalized_pagerank(damping = damping, reset = reset_vector))
        return personalized_page_rank
    
    def get_vertex_idx(self, vertex : str):
        return self.vertex_id_gen[vertex]
    
    def get_number_of_passages_by_vertex_name(self, vertex : str):
        vertex_id = self.vertex_id_gen[vertex]
        return self.graph.vs[vertex_id]['n_passages']
    
    def get_number_of_passages_by_vertex_id(self, vertex_id : int):
        return self.graph.vs[vertex_id]['n_passages']
    def get_vertices(self):
        return self.graph.vs['name']