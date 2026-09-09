
import streamlit as st

from Utils.css_loader import load_css

from Pages.home import home_page
from Pages.data_overview import data_overview_page
from Pages.visualization import visualization_page
from Pages.prediction import prediction_page


# page configuration

st.set_page_config(
    page_title="Gaming Review Sentiment Analyzer",
    layout="wide"
)


# load css

load_css()


# main title

st.title("Gaming Review Sentiment Analyzer")


# tabs

tab1, tab2, tab3, tab4 = st.tabs([
    "Home",
    "Data Overview",
    "Visualization",
    "Prediction"
])


# home

with tab1:
    home_page()


# data overview

with tab2:
    data_overview_page()


# visualization

with tab3:
    visualization_page()


# prediction

with tab4:
    prediction_page()

