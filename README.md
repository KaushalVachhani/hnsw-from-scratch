# HNSW from Scratch

I am learning nearest-neighbor search by building it one step at a time, starting with exact search and working toward HNSW.

## Phases

1. `phase_1_brute_force_knn`: exact KNN with cosine similarity or Euclidean distance.
2. `phase_2_knn_graph`: an exact KNN graph stored as an adjacency list, plus a 2D visualization.
3. `phase_3_graph_search`: greedy graph search and its local-minimum problem.
4. `phase_4_best_first_search`: priority-queue search with a configurable distance budget.
5. `phase_5_nsw`: incremental NSW construction and comparison with the Phase 2 KNN graph.

## Run it

This project uses Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m phase_1_brute_force_knn.benchmark
uv run python -m phase_2_knn_graph.visualize
uv run python -m phase_3_graph_search.experiment
uv run python -m phase_4_best_first_search.experiment
uv run python -m phase_5_nsw.experiment
```

The first benchmark reaches one million 128-dimensional vectors and uses about 500 MB of memory.

## Current results

These results use the fixed seeds and settings in each experiment:

- Exact cosine KNN over 1,000,000 vectors averages 23.96 ms per query.
- Greedy search with graph `k=10` gets Recall@1 of 0.94 with 54.34 average distance calculations, compared with 100 for brute force.
- Best-first search with graph `k=10` and budget `60` gets Recall@1 of 0.93.
- NSW with `M=10` and budget `70` gets Recall@1 of 0.92 and Recall@5 of 0.95.
- The exact Phase 2 KNN graph with the same `M=10` and budget `70` gets Recall@1 of 0.96 and Recall@5 of 0.99.

Recall measures how often approximate search finds the true nearest neighbors. Distance calculations are the main search-work measurement. A higher recall with fewer calculations is better.

The Phase 2 graph uses exact neighbors found from the full dataset. NSW builds incrementally by searching the graph that exists so far, so construction is cheaper but its graph can be less connected and search quality can be lower.

## WIP

Phase 5 is the current stopping point. Small `M` values fragment the NSW graph, so its pruning and connectivity still need work. The next steps are to improve NSW construction and then add HNSW layers.
