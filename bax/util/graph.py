import numpy as np
from typing import List


class Vertex:
    def __init__(self, index: int, position: np.array, neighbors=None):
        self.index = index
        self.position = position
        self.neighbors = [] if neighbors is None else neighbors

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


def backtrack_indices(goal: int, prev: List[int]):
    v = goal
    path = [v]
    while prev[v] is not None:
        path.append(prev[v])
        v = prev[v]
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


def make_grid(grid_size, x1_lims=(-1, 1), x2_lims=(-1, 1)):
    x1, x2 = np.meshgrid(np.linspace(*x1_lims, grid_size), np.linspace(*x2_lims, grid_size))
    positions = np.stack([x1.flatten(), x2.flatten()], axis=-1)
    n = len(positions)

    has_edge = [[False for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if ((abs(i - j) == 1) and (j % grid_size != 0)): # neighbors cardinal directions
                has_edge[i][j] = True
            elif (abs(i - j) == grid_size): # vertices on the edge of grid
                has_edge[i][j] = True
            elif (abs(j - i) == grid_size + 1) and (j % grid_size != 0): # diagonals
                has_edge[i][j] = True
            elif (abs(j - i) == grid_size - 1) and (i % grid_size != 0): # diagonals
                has_edge[i][j] = True
            else:
                has_edge[i][j] = False
    has_edge = np.array(has_edge)

    vertices = make_vertices(positions, has_edge)
    edges = make_edges(vertices)

    return positions, vertices, edges
