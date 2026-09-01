"""Visualize and summarize a two-dimensional KNN graph."""

import matplotlib.pyplot as plt
import numpy as np

from phase_2_knn_graph.knn_graph import build_knn_graph


def graph_stats(adjacency_list):
    """Print basic statistics for a graph adjacency list."""
    degrees = [len(neighbor_ids) for neighbor_ids in adjacency_list]

    print("Nodes:", len(adjacency_list))
    print("Average degree:", np.mean(degrees))
    print("Max degree:", np.max(degrees))
    print("Min degree:", np.min(degrees))
    print("Total edges:", sum(degrees))


def plot_knn_graph(
    points,
    adjacency_list,
    *,
    query=None,
    visited_node_ids=None,
    candidate_node_ids=None,
    expansion_order=None,
    result_node_id=None,
    title="KNN graph",
):
    """Plot a graph with optional best-first search-state overlays."""
    figure, axes = plt.subplots(figsize=(9, 7))

    for point_id, point in enumerate(points):
        x, y = point

        for neighbor_id in adjacency_list[point_id]:
            neighbor_x, neighbor_y = points[neighbor_id]
            axes.plot(
                [x, neighbor_x],
                [y, neighbor_y],
                color="lightgray",
                linewidth=0.8,
                alpha=0.4,
                zorder=1,
            )

    axes.scatter(
        points[:, 0],
        points[:, 1],
        color="tab:blue",
        s=30,
        label="Graph nodes",
        zorder=2,
    )

    if candidate_node_ids:
        candidate_indices = np.asarray(sorted(candidate_node_ids), dtype=int)
        axes.scatter(
            points[candidate_indices, 0],
            points[candidate_indices, 1],
            color="gold",
            edgecolors="black",
            marker="D",
            s=55,
            label="Candidate nodes",
            zorder=3,
        )

    if visited_node_ids:
        visited_indices = np.asarray(sorted(visited_node_ids), dtype=int)
        axes.scatter(
            points[visited_indices, 0],
            points[visited_indices, 1],
            color="tab:orange",
            s=65,
            label="Visited nodes",
            zorder=4,
        )

    if expansion_order:
        expansion_indices = np.asarray(expansion_order, dtype=int)
        expansion_points = points[expansion_indices]
        axes.scatter(
            expansion_points[:, 0],
            expansion_points[:, 1],
            facecolors="none",
            edgecolors="tab:red",
            linewidths=2,
            s=110,
            label="Expansion order",
            zorder=5,
        )

        for expansion_number, expansion_point in enumerate(
            expansion_points,
            start=1,
        ):
            axes.annotate(
                str(expansion_number),
                expansion_point,
                xytext=(5, 5),
                textcoords="offset points",
                color="darkred",
                fontsize=8,
                fontweight="bold",
                zorder=6,
            )

    if result_node_id is not None:
        result_point = points[result_node_id]
        axes.scatter(
            result_point[0],
            result_point[1],
            color="tab:green",
            edgecolors="black",
            marker="*",
            s=250,
            label="Best result",
            zorder=7,
        )

    if query is not None:
        axes.scatter(
            query[0],
            query[1],
            color="tab:purple",
            edgecolors="black",
            marker="X",
            s=150,
            label="Query",
            zorder=8,
        )

    axes.set_title(title)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.axis("equal")
    axes.grid(alpha=0.2)
    axes.legend()
    figure.tight_layout()

    return figure, axes


def main():
    """Build and visualize a reproducible example KNN graph."""
    rng = np.random.default_rng(42)

    points = rng.normal(size=(100, 2)).astype(np.float32)

    adjacency_list = build_knn_graph(points, k=3, metric="euclidean")

    graph_stats(adjacency_list)

    plot_knn_graph(points, adjacency_list)
    plt.show()


if __name__ == "__main__":
    main()