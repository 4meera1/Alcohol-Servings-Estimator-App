import streamlit as st
import pandas as pd
import joblib
import os

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="🍺 Alcohol Servings Estimator",
    page_icon="🍺",
    layout="centered",
)

# ------------------- Title & Banner -------------------
st.title("🍺 Alcohol Servings Estimator")
st.markdown(
    """
    ### Welcome!  
    This app predicts the **total litres of pure alcohol** based on the servings of beer, wine, and spirits.  
    """
)

# Banner Image (changed to use_container_width to avoid deprecation)
if os.path.exists("beer_image.jpg"):
    st.image("beer_image.jpg", use_container_width=True)

st.markdown("---")

# ------------------- Input Section -------------------
st.subheader("Enter Servings Data")

col1, col2, col3 = st.columns(3)

with col1:
    beer_servings = st.number_input(
        "🍺 Beer Servings (litres)", min_value=0, max_value=500, value=100, step=1
    )
with col2:
    spirit_servings = st.number_input(
        "🥃 Spirit Servings (litres)", min_value=0, max_value=500, value=100, step=1
    )
with col3:
    wine_servings = st.number_input(
        "🍷 Wine Servings (litres)", min_value=0, max_value=500, value=100, step=1
    )

st.markdown("---")

# ------------------- Prediction -------------------
if st.button("🔮 Predict Total Alcohol Consumption"):
    # Prepare the input data
    input_data = pd.DataFrame(
        [[beer_servings, spirit_servings, wine_servings]],
        columns=["beer_servings", "spirit_servings", "wine_servings"],
    )

    # Load the model
    try:
        model_loaded = joblib.load("linear_regression_model.pkl")
        prediction = model_loaded.predict(input_data)

        # Display result nicely
        st.success("✅ Prediction Complete!")
        st.markdown(
            f"""
            <div style="background-color:#f7fbff;padding:20px;
                        border-radius:15px;text-align:center;
                        box-shadow:2px 2px 10px rgba(0,0,0,0.15)">
                <h3 style="color:#2E86C1;margin:0;">Estimated Total Litres of Pure Alcohol</h3>
                <h1 style="color:#E67E22;margin:8px 0 0 0;">{prediction[0]:.2f} litres</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except FileNotFoundError:
        st.error(
            "⚠️ Model file 'linear_regression_model.pkl' not found. Please run **train_model.py** first."
        )
