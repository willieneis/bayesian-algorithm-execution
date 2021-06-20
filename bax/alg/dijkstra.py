"""
Dijkstra's algorithm for BAX.
"""

from argparse import Namespace
import copy
import heapq
import numpy as np

from .algorithms import Algorithm
from ..util.misc_util import dict_to_namespace
from ..util.graph import Vertex, backtrack_indices, edges_of_path, jaccard_similarity


class Dijkstra(Algorithm):
    """
    Implments the shortest path or minimum cost algorithm using the Vertex class.
    """

    def __init__(
        self,
        params=None,
        vertices=None,
        start=None,
        goal=None,
        edge_to_position=None,
        node_representation="locations",
        verbose=True,
    ):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the algorithm.
        vertices : list
            List of Vertex objects
        edge_to_position: Optional[dict].
            If node_representation is 'indices', this must map edges to positions, i.e.
            of type Dict[Tuple[int,int], np.ndarray]
        node_representation: str
            How nodes are represented. Can either be by 'locations', i.e. numpy arrays,
            or 'indices', i.e. integers
        start : Vertex
            Start vertex.
        goal : Vertex
            Goal vertex.
        verbose : bool
            If True, print description string.
        """
        super().__init__(params, verbose)

        self.vertices = vertices
        assert node_representation in ["locations", "indices"]
        if node_representation == "indices":
            assert edge_to_position is not None
        self.edge_to_position = edge_to_position
        self.node_representation = node_representation
        self.start = start
        self.goal = goal

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "Dijkstras")
        self.params.cost_func = getattr(params, "cost_func", lambda u, v: 0)
        self.params.true_cost = getattr(params, "true_cost", lambda u, v: 0)

    def initialize(self):
        """Initialize algorithm, reset execution path."""
        super().initialize()

        # Set up Dijkstra's
        self.explored = [False for _ in range(len(self.vertices))]
        self.min_cost = [float("inf") for _ in range(len(self.vertices))]
        self.prev = [None for _ in range(len(self.vertices))]
        self.to_explore = [(0, self.start)]  # initialize priority queue
        self.num_expansions = 0
        self.num_queries = 0
        self.best_cost = float("inf")
        self.best_path = []
        self.current = None
        self.curr_neigh_idx = 0
        self.do_after_query = False

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        while True:
            # Complete post-query todos
            # if self.num_queries > 0:
            if self.do_after_query:
                self.after_query()

            # If self.current exists, do next step in self.current.neighbors loop
            if self.current is not None:
                return self.get_next_edge()

            # If algorithm is already complete, return None
            if len(self.to_explore) == 0:
                if not self.best_path:
                    print("No path exists to goal")
                return None

            # If self.current does not exist, set self.current and other variables
            self.best_cost, self.current = heapq.heappop(self.to_explore)
            if self.explored[self.current.index]:
                # the same node could appear in the pqueue multiple times with different costs
                self.current = None
                continue
            self.explored[self.current.index] = True
            self.num_expansions += 1

            # If algorithm completes here, return None
            if self.current.index == self.goal.index:
                self.finish_algorithm()
                return None

            # Otherwise, run current_step()
            return self.get_next_edge()

    def get_next_edge(self):
        """Return next edge. Assumes self.current is not None."""
        self.num_queries += 1
        if self.node_representation == "locations":
            current_pos = self.current.position
            neighbor_pos = self.current.neighbors[self.curr_neigh_idx].position
            next_edge_pos = (current_pos + neighbor_pos) / 2
        elif self.node_representation == "indices":
            cur_i = self.current.index
            nei_i = self.current.neighbors[self.curr_neigh_idx].index
            next_edge_pos = self.edge_to_position[(cur_i, nei_i)]
        else:
            raise RuntimeError(
                f"Node representation {self.node_representation} not supported"
            )
        self.do_after_query = True
        return next_edge_pos

    def after_query(self):
        """To do after an edge has been queried."""
        last_edge_y = self.exe_path.y[-1]
        step_cost = np.log1p(np.exp(last_edge_y))
        assert step_cost >= 0
        neighbor = self.current.neighbors[self.curr_neigh_idx]
        self.curr_neigh_idx += 1

        if (not self.explored[neighbor.index]) and (
            self.best_cost + step_cost < self.min_cost[neighbor.index]
        ):
            # Push by cost
            heapq.heappush(self.to_explore, (self.best_cost + step_cost, neighbor))
            self.min_cost[neighbor.index] = self.best_cost + step_cost
            self.prev[neighbor.index] = self.current.index

        # Unset do_after_query
        self.do_after_query = False

        # Possibly set self.current to None and self.curr_neigh_idx back to 0
        if self.curr_neigh_idx >= len(self.current.neighbors):
            self.current = None
            self.curr_neigh_idx = 0

    def finish_algorithm(self):
        """To do if algorithm completes."""
        print(
            f"Found goal after {self.num_expansions} expansions and "
            f"{self.num_queries} queries with estimated cost {self.best_cost}"
        )
        self.best_path = [
            self.vertices[i] for i in backtrack_indices(self.current.index, self.prev)
        ]

        def print_true_cost_of_path(path):
            cost = 0
            for i in range(len(path) - 1):
                cost += self.params.true_cost(path[i], path[i + 1])[0]
            print(f"True cost: {cost}")

        print_true_cost_of_path(self.best_path)

    def run_algorithm_on_f_standalone(self, f):

        # prevent parallel processes from sharing random state
        # np.random.seed()

        def dijkstras(start: Vertex, goal: Vertex):
            """Dijkstra's algorithm."""
            explored = [False for _ in range(len(self.vertices))]
            min_cost = [float("inf") for _ in range(len(self.vertices))]
            prev = [None for _ in range(len(self.vertices))]
            to_explore = [(0, start)]  # initialize priority queue
            num_expansions = 0
            num_queries = 0

            while len(to_explore) > 0:
                best_cost, current = heapq.heappop(to_explore)
                if explored[current.index]:
                    # the same node could appear in the pqueue multiple times with different costs
                    continue
                explored[current.index] = True
                num_expansions += 1
                if current.index == goal.index:
                    print(
                        f"Found goal after {num_expansions} expansions and {num_queries} queries with estimated cost {best_cost}"
                    )
                    best_path = [
                        self.vertices[i] for i in backtrack_indices(current.index, prev)
                    ]

                    def true_cost_of_path(path):
                        cost = 0
                        for i in range(len(path) - 1):
                            cost += self.params.true_cost(path[i], path[i + 1])[0]
                        print("true cost", cost)

                    true_cost_of_path(best_path)
                    return best_cost, best_path

                for neighbor in current.neighbors:
                    num_queries += 1
                    step_cost = distance(current, neighbor)
                    if (not explored[neighbor.index]) and (
                        best_cost + step_cost < min_cost[neighbor.index]
                    ):
                        heapq.heappush(
                            to_explore, (best_cost + step_cost, neighbor)
                        )  # push by cost
                        min_cost[neighbor.index] = best_cost + step_cost
                        prev[neighbor.index] = current.index

            print("No path exists to goal")
            return float("inf"), []

        exe_path = Namespace(x=[], y=[])

        def distance(u: Vertex, v: Vertex):
            cost, xs, ys = self.params.cost_func(u, v, f)

            exe_path.x.extend(xs)
            exe_path.y.extend(ys)
            assert cost >= 0
            return cost

        min_cost = dijkstras(self.start, self.goal)

        return exe_path, min_cost

    def get_exe_path_crop(self):
        """
        Return the minimal execution path for output, i.e. cropped execution path,
        specific to this algorithm.
        """
        exe_path_crop = Namespace(x=[], y=[])

        _, best_path = self.get_output()
        for i in range(len(best_path) - 1):
            if self.node_representation == "locations":
                vec_start_pos = best_path[i].position
                vec_end_pos = best_path[i + 1].position
                edge_pos = (vec_start_pos + vec_end_pos) / 2.0
            elif self.node_representation == "indices":
                vec_start = best_path[i].index
                vec_end = best_path[i + 1].index
                edge_pos = self.edge_to_position[(vec_start, vec_end)]
            else:
                raise RuntimeError(
                    f"Node representation {self.node_representation} not supported"
                )

            exe_path_crop.x.append(edge_pos)
            idx, pos = next(
                (
                    tup
                    for tup in enumerate(self.exe_path.x)
                    if all(tup[1] == edge_pos)
                    # if np.allclose(tup[1], edge_pos)
                )
            )
            exe_path_crop.y.append(self.exe_path.y[idx])

        return exe_path_crop

    def get_copy(self):
        """Return a copy of this algorithm."""
        # Cache the three Vertex attributes
        vertices = self.vertices
        start = self.start
        goal = self.goal

        # Delete the three Vertex attributes
        del self.vertices
        del self.start
        del self.goal

        # Copy the rest of this object then define the three Vertex attributes
        algo_copy = copy.deepcopy(self)
        algo_copy.vertices = vertices
        algo_copy.start = start
        algo_copy.goal = goal

        # Re-define the three Vertex attributes in this object
        self.vertices = vertices
        self.start = start
        self.goal = goal

        # Return the copy
        return algo_copy

    def get_output(self):
        """Return best path."""
        return self.best_cost, self.best_path

    def get_output_dist_fn_path_cost(self):
        """
        Return distance function (based on difference in cost of shortest path) for
        pairs of outputs.
        """

        # Default dist_fn casts outputs to arrays and returns Euclidean distance
        def dist_fn(a, b):
            a_arr = np.array(a[0])
            b_arr = np.array(b[0])
            return np.linalg.norm(a_arr - b_arr)

        return dist_fn

    def get_output_dist_fn(self):
        """
        Return distance function (based on overlap of edges in shortest path) for pairs
        of outputs.
        """

        # Default dist_fn casts outputs to arrays and returns Euclidean distance
        def dist_fn(a, b):
            edges_a = edges_of_path(a[1])
            edges_b = edges_of_path(b[1])

            # convert to list of hashable types
            edges_a = [tuple(list(e[0]) + list(e[1])) for e in edges_a]
            edges_b = [tuple(list(e[0]) + list(e[1])) for e in edges_b]

            dist = 1 - jaccard_similarity(edges_a, edges_b)
            return dist

        return dist_fn
