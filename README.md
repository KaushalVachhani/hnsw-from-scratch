# HNSW from Scratch

This repo is a curiosity project built for fun to understand how HNSW works one step at a time. It starts with exact nearest-neighbor search, adds graph navigation, and gradually moves toward NSW and HNSW.

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

## Phases and learnings

### Phase 1: Brute-force KNN

`phase_1_brute_force_knn` compares the query with every vector using cosine similarity or Euclidean distance. This gives the exact answer, so later approximate searches can be measured against it.

The exact cosine benchmark over 1,000,000 vectors averages 23.96 ms per query. The result is accurate, but its work and memory grow directly with the dataset size. That limitation is why the next phase adds a graph.

### Phase 2: KNN graph

`phase_2_knn_graph` uses Phase 1 to find every point's exact nearest neighbors and stores those connections in an adjacency list.

Compared with Phase 1, the vectors now have a structure that can be navigated instead of always scanning the full dataset. Building this graph is still expensive because its neighbors are exact. The main learning is that graph degree and connectivity affect whether a search can reach the right area.

### Phase 3: Greedy graph search

`phase_3_graph_search` starts at one graph node and repeatedly moves to the neighbor closest to the query.

Phase 2 only builds the graph. This phase uses it for approximate search and reduces the number of distance calculations. With graph `k=10`, it gets Recall@1 of 0.94 with 54.34 average calculations, compared with 100 for brute force.

The tradeoff is that greedy search keeps only one current route. It can stop at a local minimum even when a better node exists elsewhere in the graph. That is why the next phase keeps multiple possible routes.

### Phase 4: Best-first search

`phase_4_best_first_search` adds a candidate priority queue, a visited set, and a distance-calculation budget.

Unlike greedy search, it can return to another promising candidate after exploring one node. There is no single search path, so the experiment tracks visited nodes, queued candidates, expansion order, and the best result. The budget also makes the recall versus search-work tradeoff explicit.

With graph `k=10` and budget `60`, it gets Recall@1 of 0.93. At budget `70`, Recall@1 reaches 0.96. More graph connections and a larger budget usually improve recall, but both cost more.

### Phase 5: NSW

`phase_5_nsw` builds the graph incrementally. Each new vector searches the graph built so far, connects to promising nodes, and prunes connections to the configured `M`.

This removes Phase 2's need to compute exact neighbors across the full dataset before searching. The tradeoff is that early insertion choices and pruning can produce a weaker or fragmented graph.

With `M=10` and budget `70`, NSW gets Recall@1 of 0.92 and Recall@5 of 0.95. The exact Phase 2 KNN graph gets 0.96 and 0.99 with the same settings. With `M=2`, NSW splits into 65 connected components and 51 nodes have degree zero. This shows why connectivity is central to navigable graphs and motivates adding HNSW layers.

Recall measures how often approximate search finds the true nearest neighbors. Distance calculations are the main search-work measurement. Higher recall with fewer calculations is better.

## WIP

Phase 5 is the current stopping point. Small `M` values fragment the NSW graph, so its pruning and connectivity still need work. The next steps are to improve NSW construction and then add HNSW layers.
