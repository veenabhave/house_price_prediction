import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="",
    layout="wide"
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = joblib.load("house_price_model.pkl")

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("AI-Based House Price Prediction")

st.write(
    "Enter the details of a house below to estimate its sale price "
    "using a Machine Learning model."
)

st.divider()

# --------------------------------------------------
# HOUSE DETAILS
# --------------------------------------------------

st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:

    overall_qual = st.slider(
        "Overall Quality (1–10)",
        min_value=1,
        max_value=10,
        value=7
    )

    gr_liv_area = st.number_input(
        "Living Area (sq. ft.)",
        min_value=100,
        max_value=5000,
        value=2000,
        step=50
    )

    garage_cars = st.number_input(
        "Garage Capacity (cars)",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )

    total_bsmt_sf = st.number_input(
        "Basement Area (sq. ft.)",
        min_value=0,
        max_value=5000,
        value=1500,
        step=50
    )

with col2:

    year_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2026,
        value=2010,
        step=1
    )

    full_bath = st.number_input(
        "Number of Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )

    bedroom_abv_gr = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    lot_area = st.number_input(
        "Lot Area (sq. ft.)",
        min_value=500,
        max_value=50000,
        value=8000,
        step=100
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button("Predict House Price", use_container_width=True):

    # Create input DataFrame
    new_house = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "GarageCars": [garage_cars],
        "TotalBsmtSF": [total_bsmt_sf],
        "YearBuilt": [year_built],
        "FullBath": [full_bath],
        "BedroomAbvGr": [bedroom_abv_gr],
        "LotArea": [lot_area]
    })

    # Make prediction
    predicted_price = model.predict(new_house)[0]

    # Display result
    st.success("Prediction generated successfully!")

    st.metric(
        label="Estimated House Sale Price",
        value=f"${predicted_price:,.2f}"
    )

    # Show entered values
    st.subheader("House Details Used for Prediction")

    display_data = pd.DataFrame({
        "Feature": [
            "Overall Quality",
            "Living Area",
            "Garage Capacity",
            "Basement Area",
            "Year Built",
            "Full Bathrooms",
            "Bedrooms",
            "Lot Area"
        ],
        "Value": [
            overall_qual,
            f"{gr_liv_area:,} sq. ft.",
            garage_cars,
            f"{total_bsmt_sf:,} sq. ft.",
            year_built,
            full_bath,
            bedroom_abv_gr,
            f"{lot_area:,} sq. ft."
        ]
    })

    st.table(display_data)

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Linear Regression"
    )

with col2:
    st.metric(
        "R² Score",
        "0.9724"
    )

with col3:
    st.metric(
        "MAE",
        "$10,431.66"
    )

st.write(
    "The model was trained using eight house-related features. "
    "Linear Regression achieved an R² score of 0.9724 on the selected "
    "test split."
)

# --------------------------------------------------
# FEATURES USED
# --------------------------------------------------

with st.expander("Features Used by the Model"):

    st.write("""
    The model uses the following features:

    - **OverallQual** – Overall quality of the house
    - **GrLivArea** – Above-ground living area
    - **GarageCars** – Garage capacity
    - **TotalBsmtSF** – Total basement area
    - **YearBuilt** – Year the house was built
    - **FullBath** – Number of full bathrooms
    - **BedroomAbvGr** – Number of bedrooms
    - **LotArea** – Lot area
    """)

# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.caption(
    "This application provides an ML-based estimated price using the "
    "training dataset and is intended for academic demonstration purposes."
)