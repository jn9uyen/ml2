import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import STOPWORDS, WordCloud

import visualization as viz
from embeddings import EmbeddingsClustering

OUTPUT_FOLDER = Path("data/embeddings_output")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
FIGURES_FOLDER = Path("figures")
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

viz.configure_plotting()

filename = Path("data/unique_patient_conditions_embeddings.parquet")
# filename = Path("data/patient_cleaned_conditions_embeddings.parquet")
embeddings_df = pd.read_parquet(filename)
# embeddings_df = embeddings_df.sample(n=10000, random_state=42).reset_index(drop=True)

raw_text_col = "raw_text"
cleaned_text_col = "cleaned_text"

text_index_mapping = embeddings_df[[cleaned_text_col, raw_text_col]].reset_index(
    drop=False
)


# --- Iteratively categorize text ---
# thresholds = [0.95, 0.92, 0.9, 0.87, 0.85, 0.83, 0.8]  # For testing.
start = 0.95
step = -0.03
n_thresholds = 11
thresholds = start + np.arange(n_thresholds) * step

group_relative_count_lb = 0.9
current_embeddings_df = (
    embeddings_df.drop(columns={raw_text_col})
    .drop_duplicates()
    .reset_index(drop=True)
    .copy()
)

results_summary = []

for i, threshold in enumerate(thresholds):
    print(f"{i}. Processing threshold: {threshold}...")

    embclustering = EmbeddingsClustering(
        text_col=cleaned_text_col,
        threshold=threshold,
        sample_size=10000,
        normalize_scores=False,
    )
    result_df = embclustering.transform(current_embeddings_df)
    representative_df = result_df[result_df["is_representative"] == 1]

    if i < n_thresholds - 1:
        keep_group_ids = representative_df.loc[
            representative_df["group_count_relative"] >= group_relative_count_lb,
            "group_id",
        ].tolist()
    else:
        # Keep all remaining groups.
        keep_group_ids = representative_df["group_id"].tolist()

    keep_cond = result_df["group_id"].isin(keep_group_ids)
    keep_groups_df = result_df[keep_cond].copy()

    # file_path = OUTPUT_FOLDER / f"kept_groups_threshold_{threshold}.parquet"
    # keep_groups_df.to_parquet(file_path)

    current_embeddings_df = current_embeddings_df[
        ~current_embeddings_df.index.isin(keep_groups_df.index)
    ].reset_index(drop=True)

    # Store output; update keep_group_ids to be unique over loop.
    keep_group_ids = [f"{i}_{j}" for j in keep_groups_df["group_id"].tolist()]
    keep_groups_df["group_id"] = f"{i}_" + keep_groups_df["group_id"].astype(str)

    # Squash group IDs and call it "other" for the last iteration.
    if i == n_thresholds - 1:
        keep_groups_df["group_id"] = "other"

        # Keep first is_representative = 1 only
        cond = keep_groups_df["is_representative"] == 1
        keep_groups_df["is_representative"] = ((cond.cumsum() == 1) & cond).astype(int)
        # keep_groups_df["group_count"] = keep_groups_df["group_count"].sum()

    results_summary.append(
        {
            "threshold": threshold,
            "selected_group_ids": keep_group_ids,
            "selected_df": keep_groups_df,
        }
    )

    del result_df, keep_groups_df


categorized_df = pd.concat(
    [res["selected_df"] for res in results_summary], ignore_index=True
)


# Add "raw_text" to categorized_df.
final_categorized_df = categorized_df.merge(
    text_index_mapping, on=cleaned_text_col, how="inner"
).drop_duplicates(subset=["index"])
assert len(final_categorized_df) == len(
    embeddings_df
), "Mismatch in lengths after categorization."
assert final_categorized_df["index"].is_unique, "Index is not unique after merge."

final_categorized_df["group_id"].nunique()
representative_df = final_categorized_df[final_categorized_df["is_representative"] == 1]
representative_df.shape

final_categorized_df.to_parquet("data/categorized_conditions.parquet")
representative_df.to_parquet("data/categorized_conditions_representatives.parquet")


# --- Word cloud per group ---
# Optional: Add custom words to the default stopword list
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

for group_id, group_df in categorized_df.groupby("group_id"):

    # 1. Clean each text entry to remove symbols and convert to lowercase
    # This keeps only letters and spaces.
    cleaned_texts = [
        re.sub(r"[^A-Za-z\s]", "", text.lower())
        for text in group_df[cleaned_text_col]
        if pd.notna(text)
    ]

    text = " ".join(cleaned_texts)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        max_words=100,
        stopwords=stopwords,
    ).generate(text)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud for Group {group_id}")

    viz.save_figure(fig, f"wordcloud_group_{group_id}.png", folder=FIGURES_FOLDER)
    plt.close(fig)


# # --- Single pass ---
# embclustering = EmbeddingsClustering(
#     text_col="cleaned_conditions",
#     threshold=0.95,
#     sample_size=10000,
#     normalize_scores=False,
# )
# result_df = embclustering.transform(embeddings_df)
# representative_df = result_df[result_df["is_representative"] == 1]

# result_df.to_parquet("data/patient_cleaned_conditions_embeddings_grouped.parquet")
# representative_df.to_parquet(
#     "data/patient_cleaned_conditions_embeddings_grouped_representatives.parquet"
# )


# df = pd.read_parquet("data/patient_cleaned_conditions_embeddings_grouped.parquet")

# df1 = df[df["is_representative"] == 1]
# df1["group_id"].nunique()

# df["group_id"].value_counts()
# df["group_count"] = df["group_id"].map(df["group_id"].value_counts())
# df1 = df.sort_values(
#     ["group_count", "is_representative", "group_id"], ascending=[False, False, True]
# )
# df2 = df1[df1["is_representative"] == 1]


# df[df["cleaned_conditions"] == "chronic pain"]

# cond = df["group_id"] == 755
# df.loc[cond, "group_avg_centrality"].fillna(10).value_counts()

# df.loc[cond, "cleaned_conditions"].sort_values


# --- Inspect ideal text length ---
from scipy.stats import norm

text_length_series = embeddings_df["cleaned_text"].str.len()
print(text_length_series.describe())

ideal_text_length = 25
std_dev_sigma = 10

x_values = sorted(embeddings_df["cleaned_text"].str.len())
x_values = [x for x in x_values if x <= 200]
y_values = norm.pdf(x_values, loc=ideal_text_length, scale=std_dev_sigma)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_values, y_values, label="Normal PDF")
ax.axvline(
    ideal_text_length,
    color="r",
    linestyle="--",
    label=f"Mean (μ = {ideal_text_length})",
)
# plt.show()
plt.savefig("figures/text_length_distribution.png", dpi=300)

print(1)
