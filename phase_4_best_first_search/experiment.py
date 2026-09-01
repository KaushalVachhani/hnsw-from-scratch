"""Compare best-first graph search with exact nearest-neighbor search."""

import matplotlib.pyplot as plt
import numpy as np

from phase_1_brute_force_knn.exact_knn import ExactKNN, recall_at_k
from phase_2_knn_graph.knn_graph import build_knn_graph
from phase_2_knn_graph.visualize import plot_knn_graph
from phase_4_best_first_search.best_first_search import best_first_search


def main():
    """Measure best-first search recall across reproducible KNN graphs."""
    rng = np.random.default_rng(42)

    points = rng.normal(size=(100, 2)).astype(np.float32)

    num_queries = 100
    graph_k_values = (1, 2, 3, 5, 10)
    distance_budgets = tuple(range(10, 101, 10))
    target_recall = 0.90
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

    recall_by_graph_k = {}
    adjacency_list_by_graph_k = {}
    configuration_results = []

    for graph_k in graph_k_values:
        adjacency_list = build_knn_graph(
            points,
            k=graph_k,
            metric="euclidean",
        )
        adjacency_list_by_graph_k[graph_k] = adjacency_list

        graph_recall_scores = []

        for distance_budget in distance_budgets:
            query_recall_scores = []
            distance_counts = []

            for query, exact_node_ids in zip(queries, exact_node_ids_by_query):
                (
                    best_node_id,
                    _best_distance,
                    _visited_node_ids,
                    _candidate_node_ids,
                    _expansion_order,
                    distance_count,
                    _distance_by_node_id,
                ) = best_first_search(
                    query,
                    points,
                    adjacency_list,
                    entry_point=0,
                    max_distance_calculations=distance_budget,
                )

                query_recall = recall_at_k(
                    predicted=[best_node_id],
                    ground_truth=exact_node_ids,
                    k=1,
                )

                query_recall_scores.append(query_recall)
                distance_counts.append(distance_count)

            recall_at_1 = float(np.mean(query_recall_scores))
            average_distance_count = float(np.mean(distance_counts))
            total_distance_count = sum(distance_counts)
            graph_recall_scores.append(recall_at_1)
            configuration_results.append(
                {
                    "graph_k": graph_k,
                    "distance_budget": distance_budget,
                    "recall_at_1": recall_at_1,
                    "average_distance_count": average_distance_count,
                }
            )

            print(
                f"Graph k={graph_k:>2}, "
                f"budget={distance_budget:>3}: "
                f"Recall@1={recall_at_1:.2f}, "
                f"average calculations={average_distance_count:.2f}, "
                f"total calculations={total_distance_count}"
            )

        recall_by_graph_k[graph_k] = graph_recall_scores

    eligible_configurations = [
        configuration
        for configuration in configuration_results
        if configuration["recall_at_1"] >= target_recall
    ]

    if not eligible_configurations:
        raise RuntimeError(
            f"No configuration reached target Recall@1 of {target_recall:.2f}"
        )

    best_configuration = min(
        eligible_configurations,
        key=lambda configuration: (
            configuration["average_distance_count"],
            configuration["distance_budget"],
            configuration["graph_k"],
        ),
    )

    visualization_graph_k = best_configuration["graph_k"]
    visualization_distance_budget = best_configuration["distance_budget"]
    visualization_recall = best_configuration["recall_at_1"]

    for graph_k, recall_scores in recall_by_graph_k.items():
        plt.plot(
            distance_budgets,
            recall_scores,
            marker="o",
            label=f"Graph k={graph_k}",
        )

    plt.scatter(
        visualization_distance_budget,
        visualization_recall,
        color="black",
        marker="*",
        s=180,
        label="Selected configuration",
        zorder=5,
    )

    plt.xlabel("Maximum distance-calculation budget")
    plt.ylabel("Recall@1")
    plt.title("Best-first search recall by budget and graph connectivity")
    plt.xticks(distance_budgets)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    visualization_query = queries[0]
    visualization_adjacency_list = adjacency_list_by_graph_k[
        visualization_graph_k
    ]
    (
        result_node_id,
        result_distance,
        visited_node_ids,
        candidate_node_ids,
        expansion_order,
        distance_count,
        _distance_by_node_id,
    ) = best_first_search(
        visualization_query,
        points,
        visualization_adjacency_list,
        entry_point=0,
        max_distance_calculations=visualization_distance_budget,
    )

    print("\nVisualized search:")
    print("Target Recall@1:", target_recall)
    print("Measured Recall@1:", visualization_recall)
    print("Graph k:", visualization_graph_k)
    print("Distance budget:", visualization_distance_budget)
    print("Visited nodes:", sorted(visited_node_ids))
    print("Candidate nodes:", sorted(candidate_node_ids))
    print("Expansion order:", expansion_order)
    print("Best result:", result_node_id)
    print("Best distance:", result_distance)
    print("Distance calculations:", distance_count)

    plot_knn_graph(
        points,
        visualization_adjacency_list,
        query=visualization_query,
        visited_node_ids=visited_node_ids,
        candidate_node_ids=candidate_node_ids,
        expansion_order=expansion_order,
        result_node_id=result_node_id,
        title=(
            "Best-first search "
            f"(graph k={visualization_graph_k}, "
            f"budget={visualization_distance_budget})"
        ),
    )

    plt.show()


if __name__ == "__main__":
    main()