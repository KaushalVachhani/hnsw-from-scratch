import numpy as np


class ExactKNN:
    """Brute-force k-nearest neighbors search."""

    def __init__(self, vectors: np.ndarray):
        """Initialize the index with a two-dimensional array of vectors."""
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D array")

        self.vectors = vectors
        self.norms = np.linalg.norm(vectors, axis=1)

    def search(
        self,
        query: np.ndarray,
        k: int,
        metric: str = "cosine",
    ):
        """Return the indices and values of the k nearest vectors."""
        query = np.asarray(query, dtype=np.float32)

        if query.ndim != 1:
            raise ValueError("Query must be a 1D vector")

        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError("Query dimension does not match vectors")

        if not 0 < k <= len(self.vectors):
            raise ValueError("k must be between 1 and number of vectors")

        if metric == "cosine":
            scores = self._cosine_similarity(query)

            # Larger cosine similarity = better results
            indices = np.argpartition(-scores, k - 1)[:k]

            # Sort only the selected top-k results
            indices = indices[np.argsort(-scores[indices])]

            return indices, scores[indices]

        if metric == "euclidean":
            distances = self._euclidean_distance(query)

            # Smaller distance = better
            indices = np.argpartition(distances, k - 1)[:k]

            # Sort only the selected top-k results
            indices = indices[np.argsort(distances[indices])]

            return indices, distances[indices]

        raise ValueError("metric must be 'cosine' or 'euclidean'")

    def _cosine_similarity(self, query):
        """Calculate cosine similarity between the query and every vector."""
        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            raise ValueError("Query cannot be a zero vector")

        denominator = self.norms * query_norm

        scores = np.zeros(len(self.vectors), dtype=np.float32)

        valid = denominator > 0

        scores[valid] = (
            self.vectors[valid] @ query
        ) / denominator[valid]

        return scores

    def _euclidean_distance(self, query):
        """Calculate Euclidean distance from the query to every vector."""
        diff = self.vectors - query
        return np.linalg.norm(diff, axis=1)


def recall_at_k(predicted, ground_truth, k=None):
    """Calculate recall at k: |predicted ∩ ground truth| / k."""
    predicted = list(predicted)
    ground_truth = list(ground_truth)

    if k is None:
        k = len(ground_truth)

    predicted = set(predicted[:k])
    ground_truth = set(ground_truth[:k])

    return len(predicted & ground_truth) / k


if __name__ == "__main__":
    vectors = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [5, 5],
            [-1, 0],
        ],
        dtype=np.float32,
    )

    index = ExactKNN(vectors)

    query = np.array([1, 0], dtype=np.float32)

    # Cosine similarity
    neighbor_ids, cosine_scores = index.search(
        query,
        k=3,
        metric="cosine",
    )

    print("Cosine:")
    print("IDs:", neighbor_ids)
    print("Scores:", cosine_scores)

    # Euclidean distance
    neighbor_ids, distances = index.search(
        query,
        k=3,
        metric="euclidean",
    )

    print("\nEuclidean:")
    print("IDs:", neighbor_ids)
    print("Distances:", distances)