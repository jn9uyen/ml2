"""
Traditional clustering on text embeddings.
"""

import os
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

filename = Path("data/patient_cleaned_conditions_embeddings.parquet")
data = pd.read_parquet(filename)

data = data.sample(n=10000, random_state=42).reset_index(drop=True)

text_col = "cleaned_conditions"
embeddings = data.drop(columns=[text_col]).to_numpy()

data.insert(1, "text_length", data[text_col].str.len())

# --- HDBSCAN Clustering ---
if False:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,  # Min docs needed to form a cluster
        metric="euclidean",
        cluster_selection_method="eom",
    )

    cluster_labels = clusterer.fit_predict(embeddings)
    data["hdbscan_cluster_id"] = cluster_labels

    # The number of clusters found is max(cluster_labels) + 1
    num_found_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"HDBSCAN found {num_found_clusters} clusters.")

    print(1)


# --- KMeans ---
if True:
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    num_clusters = 20
    kmeans_large = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans_large.fit(embeddings)

    data.insert(1, "group_id", kmeans_large.labels_)
    data.insert(1, "group_count", data["group_id"].map(data["group_id"].value_counts()))

    data = data.sort_values(
        ["group_count", "group_id", "text_length"], ascending=[False, True, True]
    )

    num_found_clusters = len(set(kmeans_large.labels_))
    print(f"KMeans found {num_found_clusters} clusters.")

    print(1)
