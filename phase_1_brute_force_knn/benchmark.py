"""Benchmark brute-force exact KNN performance.

Example results:

Exact KNN benchmark
================================================================================
 vectors  dimension metric  k avg_latency_ms p95_latency_ms peak_memory_mb raw_vectors_mb
   10000        128 cosine 10           0.29           0.33            5.1            4.9
  100000        128 cosine 10           2.62           2.96           50.8           48.8
 1000000        128 cosine 10          25.64          29.27          508.3          488.3
"""

import gc
import time
import tracemalloc

import numpy as np
import pandas as pd

from phase_1_brute_force_knn.exact_knn import ExactKNN


def benchmark(
    sizes=(10_000, 100_000, 1_000_000),
    dimension=128,
    k=10,
    num_queries=10,
    metric="cosine",
    seed=42,
):
    """Measure exact KNN latency and memory across dataset sizes."""
    rng = np.random.default_rng(seed)

    records = []

    for num_vectors in sizes:
        print(f"Benchmarking {num_vectors:,} vectors...")

        # Reproducible float32 dataset
        vectors = rng.normal(
            size=(num_vectors, dimension),
        ).astype(np.float32)

        # Pick reproducible queries from the dataset
        query_indices = rng.choice(
            num_vectors,
            size=num_queries,
            replace=False,
        )

        queries = vectors[query_indices]

        # Force cleanup before measuring
        gc.collect()

        tracemalloc.start()

        # Build exact index
        index = ExactKNN(vectors)

        # Warm-up
        index.search(
            queries[0],
            k=k,
            metric=metric,
        )

        latencies = []

        for query in queries:
            start = time.perf_counter()

            index.search(
                query,
                k=k,
                metric=metric,
            )

            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)

        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        records.append(
            {
                "vectors": num_vectors,
                "dimension": dimension,
                "metric": metric,
                "k": k,
                "avg_latency_ms": np.mean(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "peak_memory_mb": peak / (1024**2),
                "raw_vectors_mb": vectors.nbytes / (1024**2),
            },
        )

        del index
        del vectors
        del queries

        gc.collect()

    return pd.DataFrame(records)


if __name__ == "__main__":
    results_df = benchmark(
        sizes=[
            10_000,
            100_000,
            1_000_000,
        ],
        dimension=128,
        k=10,
        num_queries=10,
        metric="cosine",
        seed=42,
    )

    print("\nExact KNN benchmark")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "avg_latency_ms": "{:.2f}".format,
                "p95_latency_ms": "{:.2f}".format,
                "peak_memory_mb": "{:.1f}".format,
                "raw_vectors_mb": "{:.1f}".format,
            },
        )
    )
