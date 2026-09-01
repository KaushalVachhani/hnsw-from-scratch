"""Compare greedy graph search with exact nearest-neighbor search."""

import numpy as np

from phase_1_brute_force_knn.exact_knn import ExactKNN, recall_at_k
from phase_2_knn_graph.knn_graph import build_knn_graph
from phase_3_graph_search.greedy_search import greedy_search


def main():
    """Build a reproducible KNN graph and run a greedy search."""
    rng = np.random.default_rng(42)

    points = rng.normal(size=(100, 2)).astype(np.float32)

    num_queries = 100
    queries = rng.normal(
        size=(num_queries, points.shape[1]),
    ).astype(np.float32)

    exact_index = ExactKNN(points)
    exact_node_ids_by_query = [
        exact_index.search(
            query,
            k=1,
            metric="euclidean",
        )[0]
        for query in queries
    ]

    print("Queries:", num_queries)
    print(
        "Brute-force average distance calculations:",
        len(points),
    )
    print(
        "Brute-force total distance calculations:",
        num_queries * len(points),
    )

    for graph_k in (1, 2, 3, 5, 10):
        adjacency_list = build_knn_graph(
            points,
            k=graph_k,
            metric="euclidean",
        )

        recalls = []
        greedy_distance_counts = []

        for query, exact_node_ids in zip(queries, exact_node_ids_by_query):
            (
                greedy_node_id,
                _greedy_distance,
                _traversal_path,
                distance_count,
            ) = greedy_search(
                query,
                points,
                adjacency_list,
                entry_point=0,
            )

            query_recall = recall_at_k(
                predicted=[greedy_node_id],
                ground_truth=exact_node_ids,
                k=1,
            )

            recalls.append(query_recall)
            greedy_distance_counts.append(distance_count)

        print(f"\nGraph k: {graph_k}")
        print("Recall@1:", np.mean(recalls))
        print(
            "Average greedy distance calculations:",
            np.mean(greedy_distance_counts),
        )
        print(
            "Total greedy distance calculations:",
            sum(greedy_distance_counts),
        )


if __name__ == "__main__":
    main()