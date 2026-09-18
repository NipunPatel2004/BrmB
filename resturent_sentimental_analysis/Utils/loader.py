import pickle
import sys

import streamlit as st

from Utils.text_preprocessor import TextPreprocessor


@st.cache_resource
def load_pipeline():

    # The pipeline was pickled while TextPreprocessor lived in the
    # training notebook's __main__ module, so pickle looks for
    # `__main__.TextPreprocessor` when loading. Register it there first.
    sys.modules["__main__"].TextPreprocessor = TextPreprocessor

    with open(
        "Models/sentiment_pipeline.pkl",
        "rb"
    ) as file:

        pipeline = pickle.load(file)

    return pipeline


@st.cache_resource
def load_metrics():

    with open(
        "Models/metrics.pkl",
        "rb"
    ) as file:

        metrics = pickle.load(file)

    return metrics


@st.cache_resource
def load_labels():

    with open(
        "Models/label_mapping.pkl",
        "rb"
    ) as file:

        labels = pickle.load(file)

    return labels


def load_css(path="Style/style.css"):
    """
    Reads Style/style.css and injects it into the page.
    Call this once near the top of app.py and every page in Pages/.
    """

    with open(path) as file:
        css = file.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )


def style_fig(fig):
    """
    Forces every Plotly chart onto a light background with dark text,
    so charts never turn black regardless of the browser/OS theme.
    Call this on a figure right before st.plotly_chart(fig, ...).
    """

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font_color="#3B3B3B",
        title_font_color="#3B3B3B",
        legend_font_color="#3B3B3B"
    )

    fig.update_xaxes(
        color="#3B3B3B",
        gridcolor="#E0E2E6",
        linecolor="#E0E2E6"
    )

    fig.update_yaxes(
        color="#3B3B3B",
        gridcolor="#E0E2E6",
        linecolor="#E0E2E6"
    )

    return fig