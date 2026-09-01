"""Search a nearest-neighbor graph using best-first traversal."""

import heapq

import numpy as np


def distance(query, vector):
    """Calculate Euclidean distance between two vectors."""
    diff = query - vector
    return np.linalg.norm(diff)


def best_first_search(
    query,
    points,
    adjacency_list,
    entry_point,
    max_distance_calculations,
):
    """Return the best result, search state, and evaluated distances."""
    if max_distance_calculations < 1:
        raise ValueError("max_distance_calculations must be at least 1")

    entry_distance = distance(query, points[entry_point])
    distance_calculations = 1

    candidate_queue = [(entry_distance, entry_point)]
    visited_node_ids = set()
    expansion_order = []
    distance_by_node_id = {entry_point: entry_distance}

    best_node_id = entry_point
    best_distance = entry_distance

    while candidate_queue:
        _candidate_distance, candidate_node_id = heapq.heappop(candidate_queue)

        if candidate_node_id in visited_node_ids:
            continue

        visited_node_ids.add(candidate_node_id)
        expansion_order.append(candidate_node_id)

        if distance_calculations >= max_distance_calculations:
            break

        for neighbor_id in adjacency_list[candidate_node_id]:
            if neighbor_id in visited_node_ids:
                continue

            neighbor_distance = distance(query, points[neighbor_id])
            distance_calculations += 1
            distance_by_node_id[neighbor_id] = neighbor_distance

            heapq.heappush(
                candidate_queue,
                (neighbor_distance, neighbor_id),
            )

            if neighbor_distance < best_distance:
                best_node_id = neighbor_id
                best_distance = neighbor_distance

            if distance_calculations >= max_distance_calculations:
                break

        if distance_calculations >= max_distance_calculations:
            break

    candidate_node_ids = {
        node_id
        for _distance, node_id in candidate_queue
        if node_id not in visited_node_ids
    }

    return (
        best_node_id,
        best_distance,
        visited_node_ids,
        candidate_node_ids,
        expansion_order,
        distance_calculations,
        distance_by_node_id,
    )