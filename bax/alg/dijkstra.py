"""
Dijkstra's algorithm for BAX.
"""

from argparse import Namespace
import copy
import heapq
import numpy as np

from .algorithms_new import Algorithm
from ..util.misc_util import dict_to_namespace
from ..util.graph import Vertex, backtrack_indices


class Dijkstra(Algorithm):
    """
    Implments the shortest path or minimum cost algorithm using the Vertex class.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "Dijkstras")
        self.params.start = getattr(params, "start", None)
        self.params.goal = getattr(params, "goal", None)
        self.params.vertices = getattr(params, "vertices", None)
        self.params.cost_func = getattr(params, "cost_func", lambda u, v: 0)
        self.params.true_cost = getattr(params, "true_cost", lambda u, v: 0)

    def initialize(self):
        """Initialize algorithm, reset execution path."""
        super().initialize()

        # Set up Dijkstra's
        self.explored = [False for _ in range(len(self.params.vertices))]
        self.min_cost = [float("inf") for _ in range(len(self.params.vertices))]
        self.prev = [None for _ in range(len(self.params.vertices))]
        self.to_explore = [(0, self.params.start)]  # initialize priority queue
        self.num_expansions = 0
        self.num_queries = 0
        self.best_cost = float("inf")
        self.best_path = []
        self.current = None
        self.do_after_query = False

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        while True:
            # Complete post-query todos
            #if self.num_queries > 0:
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
            self.current = copy.deepcopy(self.current) # is this necessary?
            if self.explored[self.current.index]:
                # the same node could appear in the pqueue multiple times with different costs
                self.current = None
                continue
            self.explored[self.current.index] = True
            self.num_expansions += 1

            # If algorithm completes here, return None
            if self.current.index == self.params.goal.index:
                self.finish_algorithm()
                return None

            # Otherwise, run current_step()
            return self.get_next_edge()

    def get_next_edge(self):
        """Return next edge. Assumes self.current is not None."""
        self.num_queries += 1
        next_edge = (self.current.position + self.current.neighbors[0].position) / 2
        self.do_after_query = True
        return next_edge

    def after_query(self):
        """To do after an edge has been queried."""
        last_edge_y = self.exe_path.y[-1]
        step_cost = np.log1p(np.exp(last_edge_y))
        assert step_cost >= 0
        neighbor = self.current.neighbors[0]

        if (not self.explored[neighbor.index]) and (
            self.best_cost + step_cost < self.min_cost[neighbor.index]
        ):
            # Push by cost
            heapq.heappush(
                self.to_explore, (self.best_cost + step_cost, neighbor)
            )
            self.min_cost[neighbor.index] = self.best_cost + step_cost
            self.prev[neighbor.index] = self.current.index

        # Unset do_after_query
        self.do_after_query = False

        # Remove self.current.neighbors[0] and possibly set self.current to None
        del self.current.neighbors[0]
        if not self.current.neighbors:
            self.current = None

    def finish_algorithm(self):
        """To do if algorithm completes."""
        print(
            f"Found goal after {self.num_expansions} expansions and "
            f"{self.num_queries} queries with estimated cost {self.best_cost}"
        )
        self.best_path = [
            self.params.vertices[i]
            for i in backtrack_indices(self.current.index, self.prev)
        ]

        def print_true_cost_of_path(path):
            cost = 0
            for i in range(len(path) - 1):
                cost += self.params.true_cost(path[i], path[i + 1])[0]
            print(f"True cost: {cost}")

        print_true_cost_of_path(self.best_path)

    def run_algorithm_on_f_standalone(self, f):

        # prevent parallel processes from sharing random state
        #np.random.seed()

        def dijkstras(start: Vertex, goal: Vertex):
            """Dijkstra's algorithm."""
            explored = [False for _ in range(len(self.params.vertices))]
            min_cost = [float("inf") for _ in range(len(self.params.vertices))]
            prev = [None for _ in range(len(self.params.vertices))]
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
                        self.params.vertices[i]
                        for i in backtrack_indices(current.index, prev)
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

        min_cost = dijkstras(self.params.start, self.params.goal)

        return exe_path, min_cost

    def get_output(self):
        """Return best path."""
        return self.best_cost, self.best_path
        #raise RuntimeError("Can't return output from execution path for Dijkstras")

    def set_print_params(self):
        """Set self.print_params."""
        super().set_print_params()
        delattr(self.print_params, "vertices")
