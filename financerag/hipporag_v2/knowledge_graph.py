import igraph as ig
from igraph import UniqueIdGenerator
from typing import List, Union, Set
import numpy as np


class KG:
    def __init__(self, is_directed = True):
        self.graph = ig.Graph(directed=is_directed)
        self.vertex_id_gen = UniqueIdGenerator(initial = list(self.graph.vs))

    def add_vertices(self, vertices : Union[List[str]|Set[str]]):
        for vertex in vertices:
            vertex_id =  self.vertex_id_gen[vertex]
            if (vertex_id == self.graph.vcount()):
                self.graph.add_vertex(vertex)

    def add_edges(self, edges : List[List[str]]):
        vertices = set(self.graph.vs['name'])
        for edge in edges:
            if (len(edge) == 3):
                if (edge[0] not in vertices and edge[2] not in vertices): continue # Entiti
                edge_id = self.graph.get_eid(edge[0], edge[2])
                if edge_id == -1:
                    self.graph.add_edge(edge[0], edge[2])

    def run_personalized_page_rank(self, reset_vector = None, damping = 0.5):
        personalized_page_rank = np.array(self.graph.personalized_pagerank(damping = damping, resset = reset_vector))
        return personalized_page_rank

    def get_vertex_id(self, vertex):
        return self.vertex_id_gen[vertex]


    def get_number_of_vertices(self):
        return len(self.graph.vs['name'])