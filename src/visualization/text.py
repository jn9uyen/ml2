import matplotlib.pyplot as plt
from wordcloud import WordCloud

from .base import HEIGHT, WIDTH, save_figure


def plot_wordcloud(
    word_weights: dict[str, float],
    title: str,
    saveas_filename: str | None = None,
    **kwargs
):
    """
    Generate and display a word cloud from a dictionary of word weights.

    Parameters
    ----------
    word_weights : dict[str, float]
        A dictionary where keys are words and values are their importance/weight.
    title : str
        The title to display above the word cloud.
    """
    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        colormap="viridis",
    ).generate_from_frequencies(word_weights)

    fig, ax = plt.subplots(figsize=(HEIGHT, WIDTH))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    save_figure(fig, saveas_filename or "wordcloud.png", **kwargs)
    plt.close(fig)
