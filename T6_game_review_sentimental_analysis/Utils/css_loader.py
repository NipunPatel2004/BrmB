
import streamlit as st
from pathlib import Path


def load_css():

    css_file = Path(__file__).parent.parent / "Style" / "style.css"

    with open(css_file, "r", encoding="utf-8") as file:
        css = file.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )