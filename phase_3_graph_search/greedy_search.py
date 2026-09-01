"""Search a nearest-neighbor graph greedily."""

import numpy as np


def distance(query, vector):
    """Calculate Euclidean distance between two vectors."""
    diff = query - vector
    return np.linalg.norm(diff)


def greedy_search(
    query,
    points,
    adjacency_list,
    entry_point,
):
    """Return the closest node found by greedy graph traversal."""
    current_node_id = entry_point
    traversal_path = [current_node_id]
    distance_calculations = 0

    while True:
        current_distance = distance(query, points[current_node_id])
        distance_calculations += 1
        best_neighbor_id = current_node_id

        for neighbor_id in adjacency_list[current_node_id]:
            neighbor_distance = distance(query, points[neighbor_id])
            distance_calculations += 1

            if neighbor_distance < current_distance:
                best_neighbor_id = neighbor_id
                current_distance = neighbor_distance

        if best_neighbor_id == current_node_id:
            break

        current_node_id = best_neighbor_id
        traversal_path.append(current_node_id)

    return current_node_id, current_distance, traversal_path, distance_calculations
