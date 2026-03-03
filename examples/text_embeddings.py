import os
import random

import hdbscan
import numpy as np
import pandas as pd
from faker import Faker
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import utils
import visualization as viz

project_root = utils.get_project_root()

MODELS_FOLDER = project_root / "models"

# --- Generate Sample Text Data ---
fake = Faker()

topics = {
    "Business & Finance": [
        lambda: fake.bs() + ". " + fake.catch_phrase() for _ in range(200)
    ],
    "Technology & Web": [
        lambda: fake.sentence(nb_words=12)
        + f" using {random.choice(['Python', 'React', 'Docker'])}."
        for _ in range(200)
    ],
    "Health & Wellness": [
        lambda: f"The study on {fake.word()} showed benefits for "
        f"{random.choice(['cardiovascular health', 'mental clarity', 'joint mobility'])}."
        for _ in range(200)
    ],
    "Travel & Leisure": [
        lambda: f"My trip to {fake.city()} was amazing, especially the "
        f"{random.choice(['local markets', 'historic museums', 'beautiful parks'])}."
        for _ in range(200)
    ],
    "Legal & Contracts": [
        lambda: f"The clause regarding {fake.word()} in the agreement specifies terms for "
        f"{random.choice(['termination', 'liability', 'confidentiality'])}."
        for _ in range(200)
    ],
}

all_texts = []
for category, generators in topics.items():
    all_texts.extend([gen() for gen in generators])

# random.shuffle(all_texts)  # Mix up the data
df = pd.DataFrame(all_texts, columns=["text"])
df["text"] = df["text"].astype(str)
print(f"Generated text df with {len(df)} rows.")


# --- Text Embeddings ---
model_name = "all-MiniLM-L6-v2"
local_model_path = MODELS_FOLDER / "all-MiniLM-L6-v2-local"

# Check if the model exists locally. If not, download and save it.
if not os.path.exists(local_model_path):
    print(f"Model '{model_name}' not found at '{local_model_path}'.")
    print("Downloading and saving model...")

    # Download the model from Hugging Face
    model = SentenceTransformer(model_name)

    # Save the model to the specified local path
    model.save(str(local_model_path))
    print(f"Model saved to '{local_model_path}'.")
else:
    print(f"Model found locally at '{local_model_path}'.")

# Load the model from the local path
print("Loading model from disk...")
model = SentenceTransformer(str(local_model_path))

# Generate embeddings for the text data
print("Generating text embeddings...")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

# --- Semantic Clustering of Text Embeddings ---
num_clusters = 5  # Hyperparameter: Here, we know we created 5 topics.
kmeans_large = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans_large.fit(embeddings)

df["kmeans_cluster_id"] = kmeans_large.labels_


# --- HDBSCAN Clustering ---
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,  # Min docs needed to form a cluster
    metric="euclidean",
    cluster_selection_method="eom",
)

cluster_labels = clusterer.fit_predict(embeddings)
df["hdbscan_cluster_id"] = cluster_labels

# The number of clusters found is max(cluster_labels) + 1
num_found_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"HDBSCAN found {num_found_clusters} clusters.")

# --- Latent Dirichlet Allocation (LDA) ---
"""
Traditional Topic Modeling
Latent Dirichlet Allocation (LDA) is a classic method that works on word counts,
not embeddings. It assumes documents are a mixture of topics, and topics are a
mixture of words.

Pros: Highly interpretable; topics are defined by a list of their most important words.

Cons: Doesn't understand semantic similarity (e.g., "car" and "automobile" are treated
as completely different words). It often requires more text preprocessing
(like stop-word removal).
"""
# Create a document-term matrix (X)
vectorizer = CountVectorizer(stop_words="english", max_features=500)
X = vectorizer.fit_transform(df["text"])

# Run LDA
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(X)

# Get the topic distribution for each document: shape (n_samples, n_topics)
# For each document, it gives the probability of each topic.
topic_distributions = lda_model.transform(X)
assert all(
    abs(topic_distributions.sum(axis=1) - 1) < 1e-6
), "Each document's topic distribution should sum to 1."


def print_top_words(model, feature_names, n_top_words):
    """
    Print the top `n_top_words` words for each topic in the LDA model.
    """
    for topic_idx, topic in enumerate(model.components_):
        message = f"Topic #{topic_idx}: "
        message += " ".join(
            [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
        )
        print(message)


print_top_words(lda_model, vectorizer.get_feature_names_out(), 10)


# --- Add topic information back to text documents ---
# Get the dominant topic for each document
df["lda_dominant_topic_id"] = np.argmax(topic_distributions, axis=1)

# Get the probability or 'score' of that dominant topic
df["lda_dominant_topic_score"] = np.max(topic_distributions, axis=1)

# You can also add the full distribution for each document if needed
# This creates a column where each entry is a list/array of 5 numbers
df["lda_topic_distribution"] = list(topic_distributions)

# --- Interpreting the Clusters ---
print("\n--- Interpreting the KMeans Clusters ---")
for i in range(num_clusters):
    print(f"\n--- Cluster {i + 1} ---")
    # Taking a random sample is better for large clusters
    sample_texts = df[df["kmeans_cluster_id"] == i]["text"].sample(3).tolist()
    for text in sample_texts:
        print(f"- {text}")

# --- Interpreting the HDBSCAN Clusters ---
print("\n--- Interpreting the HDBSCAN Clusters ---")
for i in range(num_found_clusters):
    print(f"\n--- Cluster {i + 1} ---")
    # Taking a random sample is better for large clusters
    sample_texts = df[df["hdbscan_cluster_id"] == i]["text"].sample(3).tolist()
    for text in sample_texts:
        print(f"- {text}")


# Word Cloud Visualization
feature_names = vectorizer.get_feature_names_out()
n_top_words = 30

for topic_idx, topic in enumerate(lda_model.components_):
    # Prepare the word_weights dictionary for the current topic
    top_word_indices = topic.argsort()[: -n_top_words - 1 : -1]
    word_weights = {str(feature_names[i]): topic[i] for i in top_word_indices}

    plot_title = f"Word Cloud for Topic #{topic_idx}"
    saveas_filename = f"wordcloud_topic_{topic_idx}.png"
    viz.text.plot_wordcloud(word_weights, plot_title, saveas_filename=saveas_filename)


print(2)
