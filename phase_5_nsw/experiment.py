"""Compare incremental NSW with the offline Phase 2 KNN graph."""

import heapq

import matplotlib.pyplot as plt
import numpy as np

from phase_1_brute_force_knn.exact_knn import ExactKNN, recall_at_k
from phase_2_knn_graph.knn_graph import build_knn_graph
from phase_4_best_first_search.best_first_search import best_first_search
from phase_5_nsw.nsw import NSW


def average_graph_degree(adjacency_list):
    """Return the average number of outgoing neighbors per node."""
    if not adjacency_list:
        return 0.0

    return float(
        np.mean([len(neighbor_ids) for neighbor_ids in adjacency_list])
    )


def graph_statistics(adjacency_list):
    """Return degree metrics and weakly connected component count."""
    if not adjacency_list:
        return {
            "connected_components": 0,
            "degree_zero_nodes": 0,
            "minimum_degree": 0,
            "maximum_degree": 0,
            "average_degree": 0.0,
        }

    degrees = np.asarray(
        [len(neighbor_ids) for neighbor_ids in adjacency_list],
        dtype=int,
    )
    undirected_adjacency = [
        set(neighbor_ids)
        for neighbor_ids in adjacency_list
    ]

    for node_id, neighbor_ids in enumerate(adjacency_list):
        for neighbor_id in neighbor_ids:
            undirected_adjacency[neighbor_id].add(node_id)

    unvisited_node_ids = set(range(len(adjacency_list)))
    connected_components = 0

    while unvisited_node_ids:
        connected_components += 1
        component_stack = [unvisited_node_ids.pop()]

        while component_stack:
            node_id = component_stack.pop()

            for neighbor_id in undirected_adjacency[node_id]:
                if neighbor_id not in unvisited_node_ids:
                    continue

                unvisited_node_ids.remove(neighbor_id)
                component_stack.append(neighbor_id)

    return {
        "connected_components": connected_components,
        "degree_zero_nodes": int(np.sum(degrees == 0)),
        "minimum_degree": int(np.min(degrees)),
        "maximum_degree": int(np.max(degrees)),
        "average_degree": float(np.mean(degrees)),
    }


def evaluate_nsw(index, queries, exact_ids_by_query, search_budget):
    """Measure NSW recall and average query distance calculations."""
    recall_at_1_scores = []
    recall_at_5_scores = []
    distance_counts = []

    for query, exact_ids in zip(queries, exact_ids_by_query):
        approximate_ids, _distances, distance_count = index.search(
            query,
            k=5,
            budget=search_budget,
        )
        recall_at_1_scores.append(
            recall_at_k(approximate_ids, exact_ids, k=1)
        )
        recall_at_5_scores.append(
            recall_at_k(approximate_ids, exact_ids, k=5)
        )
        distance_counts.append(distance_count)

    return {
        "recall_at_1": float(np.mean(recall_at_1_scores)),
        "recall_at_5": float(np.mean(recall_at_5_scores)),
        "average_distance_calculations": float(np.mean(distance_counts)),
        "average_graph_degree": average_graph_degree(
            index.adjacency_list
        ),
    }


def evaluate_knn_graph(
    points,
    adjacency_list,
    queries,
    exact_ids_by_query,
    search_budget,
):
    """Measure Phase 2 graph recall with Phase 4 best-first search."""
    recall_at_1_scores = []
    recall_at_5_scores = []
    distance_counts = []

    for query, exact_ids in zip(queries, exact_ids_by_query):
        (
            _best_node_id,
            _best_distance,
            _visited_node_ids,
            _candidate_node_ids,
            _expansion_order,
            distance_count,
            distance_by_node_id,
        ) = best_first_search(
            query,
            points,
            adjacency_list,
            entry_point=0,
            max_distance_calculations=search_budget,
        )
        evaluated_candidates = [
            (distance, node_id)
            for node_id, distance in distance_by_node_id.items()
        ]
        nearest_candidates = heapq.nsmallest(5, evaluated_candidates)
        approximate_ids = [
            node_id
            for _distance, node_id in nearest_candidates
        ]

        recall_at_1_scores.append(
            recall_at_k(approximate_ids, exact_ids, k=1)
        )
        recall_at_5_scores.append(
            recall_at_k(approximate_ids, exact_ids, k=5)
        )
        distance_counts.append(distance_count)

    return {
        "recall_at_1": float(np.mean(recall_at_1_scores)),
        "recall_at_5": float(np.mean(recall_at_5_scores)),
        "average_distance_calculations": float(np.mean(distance_counts)),
        "average_graph_degree": average_graph_degree(adjacency_list),
    }


def print_results(results):
    """Print experiment results in aligned columns."""
    print(
        f"{'Graph':<12}"
        f"{'M':>4}"
        f"{'Budget':>9}"
        f"{'Recall@1':>12}"
        f"{'Recall@5':>12}"
        f"{'Avg calculations':>20}"
        f"{'Avg degree':>14}"
    )
    print("-" * 83)

    for result in results:
        print(
            f"{result['graph']:<12}"
            f"{result['m']:>4}"
            f"{result['search_budget']:>9}"
            f"{result['recall_at_1']:>12.2f}"
            f"{result['recall_at_5']:>12.2f}"
            f"{result['average_distance_calculations']:>20.2f}"
            f"{result['average_graph_degree']:>14.2f}"
        )


def print_graph_statistics(statistics_results):
    """Print one structural summary for each graph and M value."""
    print(
        f"{'Graph':<12}"
        f"{'M':>4}"
        f"{'Components':>13}"
        f"{'Degree 0':>11}"
        f"{'Min degree':>12}"
        f"{'Max degree':>12}"
        f"{'Avg degree':>12}"
    )
    print("-" * 76)

    for result in statistics_results:
        print(
            f"{result['graph']:<12}"
            f"{result['m']:>4}"
            f"{result['connected_components']:>13}"
            f"{result['degree_zero_nodes']:>11}"
            f"{result['minimum_degree']:>12}"
            f"{result['maximum_degree']:>12}"
            f"{result['average_degree']:>12.2f}"
        )


def plot_results(results, max_neighbor_values, search_budgets):
    """Plot Recall@1 and Recall@5 across search budgets."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    for graph_name, line_style, marker in (
        ("NSW", "-", "o"),
        ("Phase 2 KNN", "--", "s"),
    ):
        for max_neighbors in max_neighbor_values:
            graph_results = [
                result
                for result in results
                if result["graph"] == graph_name
                and result["m"] == max_neighbors
            ]
            label = f"{graph_name}, M={max_neighbors}"

            axes[0].plot(
                search_budgets,
                [result["recall_at_1"] for result in graph_results],
                linestyle=line_style,
                marker=marker,
                label=label,
            )
            axes[1].plot(
                search_budgets,
                [result["recall_at_5"] for result in graph_results],
                linestyle=line_style,
                marker=marker,
                label=label,
            )

    for axes_item, recall_label in zip(
        axes,
        ("Recall@1", "Recall@5"),
    ):
        axes_item.set_xlabel("Search distance-calculation budget")
        axes_item.set_ylabel(recall_label)
        axes_item.set_title(f"{recall_label} by graph and budget")
        axes_item.set_xticks(search_budgets)
        axes_item.set_ylim(0, 1.05)
        axes_item.grid(alpha=0.3)
        axes_item.legend()

    figure.suptitle("Incremental NSW compared with the Phase 2 KNN graph")
    figure.tight_layout()


def main():
    """Run the reproducible NSW and KNN graph comparison."""
    num_points = 100
    dimension = 2
    num_queries = 100
    seed = 42
    max_neighbor_values = (2, 5, 10)
    construction_budget = 50
    search_budgets = tuple(range(10, 101, 10))

    rng = np.random.default_rng(seed)
    points = rng.normal(
        size=(num_points, dimension),
    ).astype(np.float32)
    queries = rng.normal(
        size=(num_queries, dimension),
    ).astype(np.float32)

    exact_index = ExactKNN(points)
    exact_ids_by_query = [
        exact_index.search(
            query,
            k=5,
            metric="euclidean",
        )[0]
        for query in queries
    ]

    results = []
    statistics_results = []

    for max_neighbors in max_neighbor_values:
        nsw_index = NSW(
            max_neighbors=max_neighbors,
            construction_budget=construction_budget,
        )

        for point in points:
            nsw_index.add(point)

        knn_adjacency_list = build_knn_graph(
            points,
            k=max_neighbors,
            metric="euclidean",
        )

        for graph_name, adjacency_list in (
            ("NSW", nsw_index.adjacency_list),
            ("Phase 2 KNN", knn_adjacency_list),
        ):
            statistics = graph_statistics(adjacency_list)
            statistics.update(
                {
                    "graph": graph_name,
                    "m": max_neighbors,
                }
            )
            statistics_results.append(statistics)

        for search_budget in search_budgets:
            nsw_result = evaluate_nsw(
                nsw_index,
                queries,
                exact_ids_by_query,
                search_budget,
            )
            nsw_result.update(
                {
                    "graph": "NSW",
                    "m": max_neighbors,
                    "search_budget": search_budget,
                }
            )
            results.append(nsw_result)

            knn_result = evaluate_knn_graph(
                points,
                knn_adjacency_list,
                queries,
                exact_ids_by_query,
                search_budget,
            )
            knn_result.update(
                {
                    "graph": "Phase 2 KNN",
                    "m": max_neighbors,
                    "search_budget": search_budget,
                }
            )
            results.append(knn_result)

    print("Experiment settings:")
    print("Points:", num_points)
    print("Dimension:", dimension)
    print("Queries:", num_queries)
    print("Seed:", seed)
    print("Construction budget:", construction_budget)
    print("Search budgets:", search_budgets)

    print("\nGraph structure:")
    print_graph_statistics(statistics_results)

    print("\nSearch quality:")
    print_results(results)
    plot_results(results, max_neighbor_values, search_budgets)
    plt.show()


if __name__ == "__main__":
    main()
