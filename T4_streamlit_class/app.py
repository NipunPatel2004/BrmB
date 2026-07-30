import streamlit as st
from Utils.css_loader import load_css
from Pages.prediction import prediction_page
from Pages.overview import overview
from Pages.visualization import visualization
from Pages.home import home
load_css()


# page configration 
st.set_page_config(
    page_title="heart disase prediction",
    layout="wide"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Home", "Data Overview", "Visualization", "Prediction"]
)

with tab1:
    st.header("Home")
    home()

with tab2:
    st.header("Data Overview")
    overview()

with tab3:
    st.header("Visualization")
    visualization()

with tab4:
    st.header("Prediction")
    prediction_page()