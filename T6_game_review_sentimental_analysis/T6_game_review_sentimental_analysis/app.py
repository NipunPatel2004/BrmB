import streamlit as st

from Pages.home import home_page
from Pages.data_overview import data_overview_page

from Pages.visualization import visualization_page
from Pages.prediction import prediction_page


# Page Configuration

st.set_page_config(
    page_title="Gaming Review Sentiment Analyzer",
    layout="wide"
)


# Main Title

st.title("Gaming Review Sentiment Analyzer")

# Tabs

tab1, tab2, tab3, tab4 = st.tabs([
    "Home",
    "Data Overview",
    "Visualization",
    "Prediction"
])


# Home


with tab1:
    home_page()


# Data Overview

with tab2:
    data_overview_page()


# Visualization

with tab3:
    visualization_page()



# Prediction


with tab4:
    prediction_page()