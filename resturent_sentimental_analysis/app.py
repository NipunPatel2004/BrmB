import streamlit as st

from Utils.loader import load_css

st.set_page_config(
    page_title="Restaurant Review Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

st.markdown(
    '<div class="page-title home-title">Restaurant Review Analysis System</div>',
    unsafe_allow_html=True
)

st.markdown("""
Welcome to the Restaurant Review Analysis System.

This platform is designed to analyze customer reviews and provide meaningful insights into customer opinions and experiences. Reviews often contain valuable feedback about food quality, service, staff behavior, ambience, pricing, and overall customer satisfaction.

By exploring review data and analyzing sentiment patterns, users can better understand how customers perceive restaurant experiences and identify common themes within feedback.
""")

st.divider()

st.markdown(
    '<div class="section-title">System Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    Dataset Exploration

    1- View and explore review records

    2- Understand dataset structure

    3- Examine sentiment categories

    4- Analyze review information

    5- Explore customer feedback patterns
    """)

with col2:
    st.info("""
    Exploratory Data Analysis

    1- Review sentiment distribution

    2- Analyze frequently used words

    3- Discover common phrases

    4 Explore review length patterns

    5- Visualize review insights through charts
    """)

with col3:
    st.info("""
    Sentiment Prediction

    1- Analyze new customer reviews

    2- Predict review sentiment

    3- View confidence scores

    4- Compare positive and negative probabilities

    5- Understand influential words within reviews
    """)

st.divider()

st.markdown('<div class="section-title">Purpose of the System</div>', unsafe_allow_html=True)

st.markdown("""
Customer reviews provide valuable information that can help organizations understand customer experiences and expectations.

This system enables users to:

- Understand overall customer sentiment.
- Identify strengths highlighted by customers.
- Detect recurring concerns and complaints.
- Explore trends within customer feedback.
- Gain deeper insights from textual reviews.
- Support data-driven decision making.

Rather than relying solely on ratings, review analysis helps uncover the reasons behind customer satisfaction and dissatisfaction.
""")

st.divider()

st.markdown('<div class="section-title">Application Modules</div>', unsafe_allow_html=True)

st.markdown("""
**Home**

Provides an overview of the application, its objectives, and available features.

**Dataset**

Allows users to explore the review dataset and understand the available information.

**EDA Analysis**

Provides interactive visualizations and insights, including sentiment distribution, keyword analysis, review patterns, and phrase analysis.

**Prediction**

Allows users to enter a review and receive a sentiment prediction along with supporting insights.
""")

st.divider()

st.markdown('<div class="section-title">Benefits</div>', unsafe_allow_html=True)

st.markdown("""
- User-friendly interface
- Interactive visualizations
- Efficient review analysis
- Clear sentiment insights
- Real-time prediction capability
- Useful for academic, research, and demonstration purposes

The system is designed to transform customer reviews into meaningful information that is easy to understand and explore.
""")

st.divider()

st.markdown(
    '<div class="app-footer">Restaurant Review Analysis System</div>',
    unsafe_allow_html=True
)