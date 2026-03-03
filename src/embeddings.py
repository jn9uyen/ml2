import collections
from typing import Literal

import faiss
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import RobustScaler
from tqdm.auto import tqdm

import utils


class UnionFind:
    """A memory-efficient data structure for finding connected components."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        """Find the root parent of an item, with path compression."""
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        """Merge the sets containing i and j, by rank."""
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j
                if self.rank[root_i] == self.rank[root_j]:
                    self.rank[root_j] += 1

    def get_groups(self) -> list[list[int]]:
        """Retrieve all the disjoint sets (groups)."""
        groups = collections.defaultdict(list)
        for i in range(len(self.parent)):
            root = self.find(i)
            groups[root].append(i)
        return list(groups.values())


class EmbeddingsClustering:
    """
    A class for clustering embeddings.

    Parameters
    ----------
    text_col : str
        The name of the column containing the original text.
    threshold : float, default=0.8
        The cosine similarity threshold for grouping embeddings.
    method : {'cosine_similarity', 'faiss'}, default='faiss'
        The method to use for clustering:
        - 'cosine_similarity': Use sklearn's cosine_similarity (may be memory-intensive)
        - 'faiss': Use FAISS for memory-efficient pair finding (recommended for large
        datasets)
    ideal_text_length : int, optional, default=100
        The ideal text length of the representative member. If None, the ideal length
        is the median text length of all members in the data.
    sort_by_cohesion : bool, default=True
        If True, sort clusters from most to least cohesive based on their average
        internal similarity
    batch_size : int, default=1024
        The batch size to use for processing in _cluster_embeddings_faiss().
    sample_size : int, optional
        The number of samples to draw from the group for similarity calculation.
    random_state : int, optional
        Random seed for sampling.
    """

    def __init__(
        self,
        text_col: str,
        threshold: float = 0.8,
        method: Literal["faiss", "cosine_similarity"] = "faiss",
        ideal_text_length: int | None = 25,
        ideal_text_length_std: int | None = 10,
        sort_by_cohesion: bool = True,
        normalize_scores: bool = True,
        batch_size: int = 1024,
        sample_size: int = 1000,
        random_state: int | None = 456,
    ):
        self.text_col = text_col
        self.threshold = threshold
        self.method = method
        self.ideal_text_length = ideal_text_length
        self.ideal_text_length_std = ideal_text_length_std
        self.sort_by_cohesion = sort_by_cohesion
        self.normalize_scores = normalize_scores
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.random_state = random_state

    def transform(
        self, df: pd.DataFrame, return_embeddings: bool = False
    ) -> pd.DataFrame:
        """
        Group the embeddings in a DataFrame and label the representative embedding
        in each group.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a text column and embedding columns
            (e.g., "emb_0", "emb_1", ...).
        return_embeddings : bool, default=False
            If True, include the original embedding columns in the output.
        """
        assert (
            self.text_col in df.columns
        ), f"Column '{self.text_col}' not found in DataFrame."

        data = df.copy()
        embedding_cols = [col for col in data.columns if col not in [self.text_col]]
        embeddings = data[embedding_cols].to_numpy()
        data = data[[self.text_col]]

        # --- Text length score using Gaussian kernel centered at ideal_text_length ---
        if self.ideal_text_length is None:
            self.ideal_text_length = data[self.text_col].str.len().median()

        if self.ideal_text_length_std is None:
            self.ideal_text_length_std = data[self.text_col].str.len().std()

        data["text_length"] = data[self.text_col].str.len()
        diff = data["text_length"] - self.ideal_text_length
        scale = self.ideal_text_length_std
        data["text_length_score"] = np.exp(-(diff**2) / (2 * scale**2))

        # --- Cluster embeddings ---
        groups, graph, similarity_matrix = self._cluster_embeddings(data, embeddings)

        # --- Label representative member in each group ---
        representative_indices = []
        all_centrality_scores = []

        for group in tqdm(groups, desc="Finding representative group member"):
            group_indices = list(group)
            centrality_scores = self._compute_centrality_scores(
                group_indices,
                data,
                similarity_matrix=(
                    similarity_matrix if self.method == "cosine_similarity" else None
                ),
                embeddings=embeddings if self.method == "faiss" else None,
            )

            if self.normalize_scores:
                # Final score is weighted sum of normalized centrality scores.
                scaler = RobustScaler()
                X = centrality_scores[["member_centrality", "text_length_score"]]
                X = scaler.fit_transform(X)
                centrality_scores["final_score"] = 2 * X[:, 0] + X[:, 1]
            else:
                centrality_scores["final_score"] = (
                    centrality_scores["member_centrality"]
                    * centrality_scores["text_length_score"]
                )
            best_candidate_idx = centrality_scores["final_score"].idxmax()

            representative_indices.append(best_candidate_idx)
            all_centrality_scores.append(centrality_scores)

        # --- Assemble final DataFrame ---
        cols = ["group_avg_centrality", "member_centrality", "final_score"]
        all_centrality_scores_df = pd.concat(all_centrality_scores)
        data = data.join(all_centrality_scores_df[cols])

        # Assign group ID and representative flag.
        index_to_group_id_map = {
            member_idx: group_id
            for group_id, group_members in enumerate(groups)
            for member_idx in group_members
        }
        data["group_id"] = pd.Series(index_to_group_id_map)
        data["is_representative"] = data.index.isin(representative_indices).astype(int)
        data["group_count"] = data["group_id"].map(data["group_id"].value_counts())
        group_counts = data.groupby("group_id")["group_count"].first()
        relative, cumulative = utils.math.compute_relative_importance(group_counts)
        data["group_count_relative"] = data["group_id"].map(relative)
        data["group_count_cumulative"] = data["group_id"].map(cumulative)

        # Reorder columns and sort.
        output_cols = [
            "group_id",
            "group_count",
            "group_count_relative",
            "group_count_cumulative",
            "is_representative",
            self.text_col,
            "group_avg_centrality",
            "member_centrality",
            "text_length",
            "text_length_score",
            "final_score",
        ]
        if return_embeddings:
            output_cols += embedding_cols

        transformed_data = data[output_cols].sort_values(
            by=["group_count", "group_avg_centrality", "group_id", "is_representative"],
            ascending=[False, False, True, False],
        )
        return transformed_data

    def _cluster_embeddings(
        self, data: pd.DataFrame, embeddings: np.ndarray
    ) -> tuple[list[list[int]], nx.Graph | None, np.ndarray | None]:
        """
        Compute pairwise cosine similarity to cluster embedding vectors. Use either
        sklearn's cosine_similarity or FAISS for memory-efficient pair finding.
        """
        match self.method:
            case "cosine_similarity":
                similarity_matrix = cosine_similarity(embeddings)

                # Create a graph where an edge exists if similarity > threshold.
                adjacency_matrix = similarity_matrix > self.threshold
                graph = nx.from_numpy_array(adjacency_matrix)
                groups = list(nx.connected_components(graph))

            case "faiss":
                groups = self._cluster_embeddings_faiss(embeddings)
                graph = None
                similarity_matrix = None
            case _:
                raise ValueError(f"Unknown clustering method: {self.method}")

        if self.sort_by_cohesion:
            scores = [
                self._compute_group_cohesion(
                    list(g),
                    data,
                    similarity_matrix=similarity_matrix,
                    embeddings=embeddings if self.method == "faiss" else None,
                )
                for g in groups
            ]
            sorted_pairs = sorted(zip(groups, scores), key=lambda x: x[1], reverse=True)
            groups = [group for group, score in sorted_pairs]

        return groups, graph, similarity_matrix

    def _cluster_embeddings_faiss(self, embeddings: np.ndarray) -> list[list[int]]:
        """
        Cluster embeddings using FAISS (Facebook AI Similarity Search) for
        memory-efficient pair finding.
        """
        num_embeddings, dim = embeddings.shape

        # FAISS works with L2 distance; normalize embeddings to use Inner Product (IP)
        # as a proxy for cosine similarity. On unit vectors, IP = cosine similarity.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings: NDArray[np.float32] = (embeddings / norms).astype(
            np.float32
        )

        index = faiss.IndexFlatIP(dim)
        index.add(normalized_embeddings)  # type: ignore

        # --- Collect all edges in a simple list ---
        uf = UnionFind(num_embeddings)
        edge_count = 0

        for i in tqdm(
            range(0, num_embeddings, self.batch_size),
            desc=f"Finding similar pairs with threshold > {self.threshold}",
        ):
            end = min(i + self.batch_size, num_embeddings)
            lims, distances, indices = index.range_search(
                normalized_embeddings[i:end], self.threshold
            )  # type: ignore

            for j in range(end - i):
                query_idx = i + j
                # The indices are relative to the start of the search result.
                for neighbor_idx in indices[lims[j] : lims[j + 1]]:  # noqa: E203
                    if query_idx != neighbor_idx:
                        uf.union(query_idx, neighbor_idx)
                        edge_count += 1

        print(f"Processed {edge_count} edges.")

        groups = uf.get_groups()

        return groups

    def _get_similarity_sub_matrix(
        self,
        group_indices: list[int],
        data: pd.DataFrame,
        similarity_matrix: np.ndarray | None = None,
        embeddings: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        """
        Return a similarity sub-matrix for a group, handling sampling.

        This method is flexible: it can either extract the sub-matrix from a
        pre-computed similarity matrix or compute it on-demand from raw embeddings.
        It also handles sampling for large groups.

        Parameters
        ----------
        group_indices : list of int
            The original indices of the group members.
        similarity_matrix : np.ndarray, optional
            The full, pre-computed similarity matrix.
        embeddings : np.ndarray, optional
            The full array of embeddings for on-demand calculation.

        Returns
        -------
        tuple[np.ndarray, list[int]]
            A tuple containing:
            - The calculated similarity sub-matrix.
            - The final (potentially sampled) list of indices used.
        """
        final_indices = group_indices

        # For large groups, take a sample ordered by highest text_length_score.
        if len(final_indices) > self.sample_size:
            text_length_scores = data.loc[final_indices, "text_length_score"]
            final_indices = text_length_scores.nlargest(self.sample_size).index.tolist()

        if similarity_matrix is not None:
            # Method 1: Extract from the pre-computed full matrix.
            sub_matrix = similarity_matrix[np.ix_(final_indices, final_indices)]
        elif embeddings is not None:
            # Method 2: Compute on-demand from raw embeddings.
            group_embeddings = embeddings[final_indices]
            sub_matrix = cosine_similarity(group_embeddings)
        else:
            raise ValueError("Must provide either 'similarity_matrix' or 'embeddings'.")

        return sub_matrix, final_indices

    def _compute_group_cohesion(
        self,
        group_indices: list[int],
        data: pd.DataFrame,
        similarity_matrix: np.ndarray | None = None,
        embeddings: np.ndarray | None = None,
    ) -> float:
        """
        Calculate the average similarity for all unique pairs within a group.

        This average similarity score can be calculated either from a pre-computed
        similarity matrix or by computing similarity on-demand from raw embeddings.

        Parameters
        ----------
        group_indices : list of int
            A list of indices belonging to the group.
        similarity_matrix : np.ndarray, optional
            The full, pre-calculated pairwise similarity matrix.
        embeddings : np.ndarray, optional
            The full array of embeddings, used for on-demand calculation.

        Returns
        -------
        float
            The average intra-cluster similarity score.
        """
        if len(group_indices) < 2:
            return 0.0

        sub_matrix, final_indices = self._get_similarity_sub_matrix(
            group_indices, data, similarity_matrix, embeddings
        )

        # Sum the upper triangle (excluding the diagonal) and divide by the number of
        # pairs to get the average.
        k = len(final_indices)
        num_pairs = k * (k - 1) / 2
        if num_pairs == 0:
            return 0.0

        sum_of_pairs = sub_matrix[np.triu_indices(k, 1)].sum()
        return sum_of_pairs / num_pairs

    def _compute_centrality_scores(
        self,
        group_indices: list[int],
        data: pd.DataFrame,
        similarity_matrix: np.ndarray | None = None,
        embeddings: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Compute centrality scores for a group of text embeddings.

        Statistics:
        - Member centrality: A measure of how central a member is within the group.
        - Text length: The length of the text associated with each member.
        - Group average centrality: The average centrality score of all members in the
        group.
        """
        if len(group_indices) == 1:
            return pd.DataFrame(
                {
                    "member_centrality": [1.0],
                    "group_avg_centrality": [1.0],
                    "text_length_score": data.loc[
                        group_indices[0], "text_length_score"
                    ],
                },
                index=[group_indices[0]],
            )

        sub_matrix, final_indices = self._get_similarity_sub_matrix(
            group_indices, data, similarity_matrix, embeddings
        )

        member_centrality = (sub_matrix.sum(axis=1) - 1) / (len(final_indices) - 1)
        group_avg_centrality = member_centrality.mean()

        return pd.DataFrame(
            {
                "member_centrality": member_centrality,
                "group_avg_centrality": group_avg_centrality,
                "text_length_score": data.loc[final_indices, "text_length_score"],
            },
            index=final_indices,
        )
