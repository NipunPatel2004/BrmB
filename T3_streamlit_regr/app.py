
import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("random_forest_regressor.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page Config
st.set_page_config(
    page_title="Car Price Predictor",
    layout="centered"
)

st.title("Car Price Prediction System")
st.markdown(
    "Predict the estimated price of a car using machine learning."
)

st.divider()

st.subheader("Enter Vehicle Specifications")

# Input Fields (ranges based on dataset)

enginesize = st.number_input(
    "Engine Size",
    min_value=61,
    max_value=326,
    value=120,
    help="Engine cylinder volume. Larger engines generally provide more power. Typical range: 61–326."
)

horsepower = st.number_input(
    "Horsepower",
    min_value=48,
    max_value=288,
    value=100,
    help="Engine power output. Higher horsepower usually means better performance. Typical range: 48–288."
)

curbweight = st.number_input(
    "Curb Weight",
    min_value=1488,
    max_value=4066,
    value=2500,
    help="Vehicle weight without passengers or cargo. Typical range: 1488–4066 lbs."
)

carwidth = st.number_input(
    "Car Width",
    min_value=60.3,
    max_value=72.3,
    value=65.5,
    help="Overall width of the vehicle in inches. Typical range: 60.3–72.3."
)

wheelbase = st.number_input(
    "Wheel Base",
    min_value=86.6,
    max_value=120.9,
    value=95.0,
    help="Distance between front and rear wheels. Longer wheelbases usually improve ride comfort."
)

highwaympg = st.number_input(
    "Highway MPG",
    min_value=16,
    max_value=54,
    value=30,
    help="Fuel efficiency on highways. Higher values indicate lower fuel consumption."
)

peakrpm = st.number_input(
    "Peak RPM",
    min_value=4150,
    max_value=6600,
    value=5500,
    help="Engine speed where peak power is produced."
)

# Predict Button
if st.button("Predict Price"):

    features = np.array([[
        enginesize,
        horsepower,
        curbweight,
        carwidth,
        wheelbase,
        highwaympg,
        peakrpm
    ]])

    # Scale input
    features = scaler.transform(features)

    # Prediction
    prediction = model.predict(features)[0]

    st.divider()

    st.success(
        f" Estimated Car Price: ${prediction:,.2f}"
    )

    # Category
    st.subheader("Vehicle Category")

    if prediction < 8000:
        st.info("Budget Vehicle")

    elif prediction < 15000:
        st.info("Economy Vehicle")

    elif prediction < 25000:
        st.info("Mid-Range Vehicle")

    elif prediction < 35000:
        st.info("Premium Vehicle")

    else:
        st.info("Luxury Vehicle")

    # Performance
    st.subheader("Performance Assessment")

    if horsepower < 80:
        st.warning(
            "Low horsepower. Best suited for city driving and fuel economy."
        )

    elif horsepower < 150:
        st.info(
            "Balanced performance suitable for everyday driving."
        )

    else:
        st.success(
            "High-performance vehicle with strong acceleration."
        )

    # Fuel Efficiency
    st.subheader("Fuel Efficiency")

    if highwaympg >= 35:
        st.success(
            "Excellent fuel efficiency. Ideal for long-distance travel."
        )

    elif highwaympg >= 25:
        st.info(
            "Good fuel economy for daily use."
        )

    else:
        st.warning(
            "Lower fuel efficiency. Expect higher fuel costs."
        )

    # Comfort
    st.subheader("Comfort & Interior Space")

    if wheelbase > 105 and carwidth > 68:
        st.success(
            "Spacious cabin with excellent ride comfort."
        )

    elif wheelbase > 95:
        st.info(
            "Moderate interior space suitable for families."
        )

    else:
        st.warning(
            "Compact vehicle mainly designed for city usage."
        )

    # Engine Analysis
    st.subheader("Engine Analysis")

    if enginesize > 200:
        st.success(
            "Large engine with strong power output."
        )

    elif enginesize > 120:
        st.info(
            "Balanced engine size for performance and efficiency."
        )

    else:
        st.warning(
            "Smaller engine focused on fuel savings."
        )

    # Buyer Recommendation
    st.subheader("Buyer Recommendation")

    if prediction < 10000:
        st.write("""
        - Suitable for first-time buyers
        - Lower maintenance cost
        - Budget-friendly ownership
        - Ideal for city commuting
        """)

    elif prediction < 20000:
        st.write("""
        - Excellent value for money
        - Suitable for families
        - Balanced comfort and performance
        - Reliable daily transportation
        """)

    elif prediction < 30000:
        st.write("""
        - Premium family vehicle
        - Better comfort and safety
        - Suitable for long-distance travel
        - Strong overall driving experience
        """)

    else:
        st.write("""
        - Luxury-class vehicle
        - Premium comfort and features
        - High-performance engine
        - Executive and enthusiast buyers
        """)

    st.divider()

metrics = pickle.load(open("model_metrics.pkl", "rb"))
st.subheader(" Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("R² Score", f"{metrics['R2 Score']:.4f}")
col2.metric("MAE", f"{metrics['MAE']:.2f}")
col3.metric("MSE", f"{metrics['MSE']:.2f}")
col4.metric("RMSE", f"{metrics['RMSE']:.2f}")

