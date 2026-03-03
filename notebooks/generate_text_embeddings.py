"""
Use a text embedding model to convert text data into numerical vectors.
"""

import os
import re
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from wordcloud import STOPWORDS

DATA_FOLDER = Path("data")
DATA_FOLDER.mkdir(exist_ok=True)
MODELS_FOLDER = Path("models")
MODELS_FOLDER.mkdir(exist_ok=True)


# --- Language Model ---
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


# --- Text Data ---
text_df = pd.read_parquet(DATA_FOLDER / "unique_patient_conditions.parquet")
print(text_df.shape)
print(text_df.head(2))

text_series: pd.Series = pd.Series(text_df.dropna().squeeze())
print(text_series.shape)
print(text_series.head(2))

# text_series = text_series.head(100)

# Remove stopwords
custom_stopwords = {
    "request",
    "call",
    "called",
    "answer",
    "answered",
    "mobile",
    "email",
    "emailed",
    "phone",
    "phoned",
    "number",
    "patient",
    "please",
    "send",
    "sent",
}
stopwords = STOPWORDS.union(custom_stopwords)

cleaned_texts = [
    " ".join(
        word
        for word in re.sub(r"[^A-Za-z0-9\s]", "", text.lower()).split()
        if word not in stopwords
    )
    for text in text_series
    if pd.notna(text)
]
cleaned_texts[:5]


# --- Generate embeddings ---
generate_embeddings = True
embeddings_name = "unique_patient_conditions_embeddings.parquet"

if generate_embeddings or not os.path.exists(DATA_FOLDER / embeddings_name):
    print(f"Embeddings not found at '{DATA_FOLDER / embeddings_name}'.")
    print("Generating embeddings...")
    embeddings = model.encode(cleaned_texts, show_progress_bar=True)

    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.insert(0, "raw_text", text_series.values.tolist())
    embeddings_df.insert(1, "cleaned_text", cleaned_texts)

    embeddings_df.to_parquet(DATA_FOLDER / embeddings_name)
else:
    print(f"Embeddings found at '{DATA_FOLDER / embeddings_name}'.")
    print("Loading embeddings...")
    embeddings_df = pd.read_parquet(DATA_FOLDER / embeddings_name)


with pd.option_context("display.max_colwidth", None):
    print(embeddings_df.shape)
    print(embeddings_df.head())

print(1)
