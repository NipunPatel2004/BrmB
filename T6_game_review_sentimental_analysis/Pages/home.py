import streamlit as st

from Utils.css_loader import load_css


def home_page():

    # Load CSS
    load_css()

    # header

    st.markdown(
        """<div class="home-title">
        <h1>Gaming Review Sentiment Analyzer</h1>
        <p>Understand what gamers really think.</p>
        </div>""",
        unsafe_allow_html=True
    )

    st.divider()

    # about section

    st.subheader("About")

    st.markdown(
        """<div class="about-card">
        <h3>Turn Gaming Reviews Into Insights</h3>
        <p>
        The Gaming Review Sentiment Analyzer helps you quickly
        understand the overall opinion expressed in gaming reviews.
        Simply enter a review and discover whether the sentiment
        is <strong>Positive</strong> or <strong>Negative</strong>.
        </p>
        <p>
        The application provides a clear sentiment result along with
        prediction confidence and an easy-to-understand explanation
        of the review.
        </p>
        </div>""",
        unsafe_allow_html=True
    )

    # what you can do section

    st.subheader("What You Can Do")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """<div class="feature-card">
            <h3>Analyze Reviews</h3>
            <p>
            Enter any gaming review and instantly discover
            its overall sentiment.
            </p>
            </div>""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """<div class="feature-card">
            <h3>View Insights</h3>
            <p>
            Explore review patterns, sentiment distribution,
            review length and other useful insights.
            </p>
            </div>""",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """<div class="feature-card">
            <h3>Understand Results</h3>
            <p>
            See prediction confidence and understand the
            important parts of the review.
            </p>
            </div>""",
            unsafe_allow_html=True
        )

    # how it works section

    st.subheader("How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """<div class="step-card">
            <div class="step-number">01</div>
            <h3>Enter a Review</h3>
            <p>
            Enter or paste a gaming review into the
            prediction page.
            </p>
            </div>""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """<div class="step-card">
            <div class="step-number">02</div>
            <h3>Analyze</h3>
            <p>
            The application analyzes the review and
            determines its overall sentiment.
            </p>
            </div>""",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """<div class="step-card">
            <div class="step-number">03</div>
            <h3>Get Results</h3>
            <p>
            View the sentiment, confidence and useful
            information about your review.
            </p>
            </div>""",
            unsafe_allow_html=True
        )

    # explore the application section

    st.subheader("Explore the Application")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """<div class="explore-card">
            <h3>Review Analysis</h3>
            <p>
            Explore the available gaming review data through
            summaries and visual insights.
            </p>
            <ul>
            <li>Review overview</li>
            <li>Sentiment distribution</li>
            <li>Review length insights</li>
            <li>Word frequency</li>
            <li>Word clouds</li>
            </ul>
            </div>""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """<div class="explore-card">
            <h3>Sentiment Prediction</h3>
            <p>
            Analyze an individual gaming review and get
            an easy-to-understand result.
            </p>
            <ul>
            <li>Positive or Negative result</li>
            <li>Prediction confidence</li>
            <li>Sentiment percentage</li>
            <li>Review processing result</li>
            <li>Important words</li>
            </ul>
            </div>""",
            unsafe_allow_html=True
        )

    # start analyzing section

    st.divider()

    st.success(
        "Ready to analyze a gaming review? "
        "Go to the **Prediction** page and enter your review."
    )