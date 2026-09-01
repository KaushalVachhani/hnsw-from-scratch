"""A single-layer Navigable Small World graph index."""

import heapq

import numpy as np

from phase_4_best_first_search.best_first_search import best_first_search


class NSW:
    """Build and search a bounded-degree nearest-neighbor graph."""

    def __init__(self, max_neighbors, construction_budget=50):
        """Initialize an empty NSW index."""
        if max_neighbors < 1:
            raise ValueError("max_neighbors must be at least 1")

        if construction_budget < 1:
            raise ValueError("construction_budget must be at least 1")

        self.max_neighbors = max_neighbors
        self.construction_budget = construction_budget
        self.vectors = []
        self.adjacency_list = []
        self.entry_point = None

    def add(self, vector):
        """Add a vector, connect it to nearby nodes, and return its node ID."""
        vector = self._prepare_vector(vector)
        new_node_id = len(self.vectors)

        if self.entry_point is None:
            self.vectors.append(vector)
            self.adjacency_list.append([])
            self.entry_point = new_node_id
            return new_node_id

        evaluated_candidates, _visited_node_ids, _distance_calculations = (
            self._search_candidates(
                query=vector,
                budget=self.construction_budget,
            )
        )
        nearest_candidates = heapq.nsmallest(
            self.max_neighbors,
            evaluated_candidates,
        )
        neighbor_ids = [
            node_id
            for _distance, node_id in nearest_candidates
        ]

        self.vectors.append(vector)
        self.adjacency_list.append([])

        for neighbor_id in neighbor_ids:
            self._add_bidirectional_edge(new_node_id, neighbor_id)

        for neighbor_id in neighbor_ids:
            self._prune_neighbors(neighbor_id)

        self._prune_neighbors(new_node_id)

        return new_node_id

    def search(self, query, k, budget):
        """Return approximate nearest IDs, distances, and calculation count."""
        if self.entry_point is None:
            return (
                np.array([], dtype=int),
                np.array([], dtype=np.float32),
                0,
            )

        if not 0 < k <= len(self.vectors):
            raise ValueError("k must be between 1 and the number of vectors")

        if budget < k:
            raise ValueError("budget must be at least k")

        query = self._prepare_vector(query)
        evaluated_candidates, _visited_node_ids, distance_calculations = (
            self._search_candidates(
                query=query,
                budget=budget,
            )
        )
        nearest_candidates = heapq.nsmallest(k, evaluated_candidates)

        node_ids = np.asarray(
            [node_id for _distance, node_id in nearest_candidates],
            dtype=int,
        )
        distances = np.asarray(
            [distance for distance, _node_id in nearest_candidates],
            dtype=np.float32,
        )

        return node_ids, distances, distance_calculations

    def _search_candidates(self, query, budget):
        """Delegate candidate exploration to Phase 4 best-first search."""
        (
            _best_node_id,
            _best_distance,
            visited_node_ids,
            _candidate_node_ids,
            _expansion_order,
            distance_calculations,
            distance_by_node_id,
        ) = best_first_search(
            query,
            self.vectors,
            self.adjacency_list,
            entry_point=self.entry_point,
            max_distance_calculations=budget,
        )
        evaluated_candidates = [
            (distance, node_id)
            for node_id, distance in distance_by_node_id.items()
        ]

        return (
            evaluated_candidates,
            visited_node_ids,
            distance_calculations,
        )

    def _add_bidirectional_edge(self, first_node_id, second_node_id):
        """Connect two nodes in both adjacency lists."""
        if second_node_id not in self.adjacency_list[first_node_id]:
            self.adjacency_list[first_node_id].append(second_node_id)

        if first_node_id not in self.adjacency_list[second_node_id]:
            self.adjacency_list[second_node_id].append(first_node_id)

    def _prune_neighbors(self, node_id):
        """Keep only the closest max_neighbors links for a node."""
        neighbor_ids = self.adjacency_list[node_id]

        if len(neighbor_ids) <= self.max_neighbors:
            return

        node_vector = self.vectors[node_id]
        neighbor_ids_by_distance = sorted(
            neighbor_ids,
            key=lambda neighbor_id: self._distance(
                node_vector,
                self.vectors[neighbor_id],
            ),
        )
        retained_neighbor_ids = neighbor_ids_by_distance[
            : self.max_neighbors
        ]
        removed_neighbor_ids = set(neighbor_ids) - set(
            retained_neighbor_ids
        )

        self.adjacency_list[node_id] = retained_neighbor_ids

        for removed_neighbor_id in removed_neighbor_ids:
            reciprocal_neighbors = self.adjacency_list[
                removed_neighbor_id
            ]

            if node_id in reciprocal_neighbors:
                reciprocal_neighbors.remove(node_id)

    def _prepare_vector(self, vector):
        """Convert a vector to float32 and validate its shape."""
        vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim != 1:
            raise ValueError("vector must be one-dimensional")

        if self.vectors and vector.shape != self.vectors[0].shape:
            raise ValueError("vector dimension does not match the index")

        return vector.copy()

    @staticmethod
    def _distance(first_vector, second_vector):
        """Calculate Euclidean distance between two vectors."""
        return float(np.linalg.norm(first_vector - second_vector))