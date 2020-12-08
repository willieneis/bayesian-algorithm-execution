import numpy as np
from typing import List


class Vertex:
    def __init__(self, index: int, position: np.array, neighbors=[]):
        self.index = index
        self.position = position
        self.neighbors = neighbors

    def __repr__(self):
        return f"({self.index}, {[n.index for n in self.neighbors]})"

    def __lt__(self, other):
        return self.position[0] < other.position[0]


def make_vertices(positions: np.array, has_edge: np.array):
    n = positions.shape[0]
    vertices = [Vertex(i, p) for i, p in enumerate(positions)]
    for i in range(n):
        for j in range(i + 1, n):
            if has_edge[i, j]:
                vertices[i].neighbors.append(vertices[j])
                vertices[j].neighbors.append(vertices[i])
    return vertices


def make_edges(vertices: List[Vertex]):
    edges = []
    for v in vertices:
        for n in v.neighbors:
            edges.append((v.position, n.position))
    return edges


def farthest_pair(vertices: List[Vertex], distance_func):
    n = len(vertices)
    max_dist = -float("inf")
    pair = (None, None)
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance_func(vertices[i], vertices[j])
            if dist > max_dist:
                pair = vertices[i], vertices[j]
                max_dist = dist
    return pair


def backtrack(goal: Vertex):
    v = goal
    path = [v]
    while hasattr(v, "prev") and v.prev is not None:
        path.append(v.prev)
        v = v.prev
    return path[::-1]


def valid_path(path: List[Vertex]):
    for i in range(1, len(path)):
        if path[i] not in path[i - 1].neighbors:
            return False
    return True


def edges_of_path(path: List[Vertex]):
    assert valid_path(path)
    edges = []
    for i in range(len(path) - 1):
        edges.append((path[i].position, path[i + 1].position))
    return edges


def cost_of_path(path: List[Vertex], distance_func):
    assert valid_path(path)
    cost = 0
    for i in range(len(path) - 1):
        cost += distance_func(path[i], path[i + 1])
    return cost


def positions_of_path(path):
    positions = [v.position for v in path]
    return np.stack(positions)
