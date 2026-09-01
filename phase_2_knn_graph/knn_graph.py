"""Build an exact k-nearest-neighbor graph."""

from phase_1_brute_force_knn.exact_knn import ExactKNN


def build_knn_graph(points, k, metric="euclidean"):
    """Return a directed adjacency list of each point's nearest neighbors."""
    index = ExactKNN(points)
    adjacency_list = []

    for point_id, point in enumerate(points):
        neighbor_ids, _ = index.search(
            point,
            k=k + 1,
            metric=metric, 
        )

        neighbor_ids = neighbor_ids[neighbor_ids != point_id][:k]
        adjacency_list.append(neighbor_ids.tolist())

    return adjacency_list