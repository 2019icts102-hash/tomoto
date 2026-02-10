"""
Tomato Shelf Life Prediction Web App
Uses a trained Random Forest model to predict shelf life based on various factors.
"""

import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Tomato Shelf Life Predictor",
    page_icon="🍅",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #E63946;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #E63946 0%, #F1FAEE 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1.5rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1D3557;
    }
    .prediction-label {
        font-size: 1.2rem;
        color: #457B9D;
    }
    .info-card {
        background-color: #F1FAEE;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #E63946;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🍅 Tomato Shelf Life Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict how long your tomatoes will last based on variety, storage conditions, and transport</p>', unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "tomato_shelf_life_model.pkl")
    if not os.path.exists(model_path):
        # Try alternative paths
        alt_paths = [
            "/workspace/user_input_files/tomato_shelf_life_model.pkl",
            "tomato_shelf_life_model.pkl"
        ]
        for path in alt_paths:
            if os.path.exists(path):
                model_path = path
                break
    return joblib.load(model_path)

# Define the feature columns (must match training data)
FEATURE_COLUMNS = [
    'Maturity_Stage', 'Temp_C', 'Humidity_Pct', 'Transport_KM',
    'Packaging_Plastic Crate', 'Packaging_Polysack Bag', 'Packaging_Wooden Crate',
    'Variety_Lanka Sour', 'Variety_T-146', 'Variety_T-245', 'Variety_Thilina Tomato'
]

# Variety and Packaging options
VARIETIES = ["Lanka Sour", "T-146", "T-245", "Thilina Tomato"]
PACKAGING_TYPES = ["Plastic Crate", "Wooden Crate", "Polysack Bag"]

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load the model. Please ensure 'tomato_shelf_life_model.pkl' is in the app directory.")
    st.stop()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌱 Tomato Details")
    
    variety = st.selectbox(
        "Tomato Variety",
        options=VARIETIES,
        help="Select the variety of tomato"
    )
    
    maturity_stage = st.slider(
        "Maturity Stage",
        min_value=1,
        max_value=6,
        value=3,
        help="1 = Green, 6 = Fully Ripe"
    )
    
    # Visual indicator for maturity
    maturity_labels = {1: "🟢 Green", 2: "🟡 Breaker", 3: "🟠 Turning", 
                       4: "🟠 Pink", 5: "🔴 Light Red", 6: "🔴 Fully Ripe"}
    st.caption(f"Current: {maturity_labels.get(maturity_stage, '')}")

with col2:
    st.markdown("### 📦 Storage & Transport")
    
    packaging = st.selectbox(
        "Packaging Type",
        options=PACKAGING_TYPES,
        help="Select the packaging method"
    )
    
    transport_km = st.number_input(
        "Transport Distance (KM)",
        min_value=0,
        max_value=500,
        value=100,
        step=10,
        help="Distance from farm to storage/market"
    )

# Environmental conditions
st.markdown("### 🌡️ Environmental Conditions")
env_col1, env_col2 = st.columns(2)

with env_col1:
    temperature = st.slider(
        "Temperature (°C)",
        min_value=15,
        max_value=40,
        value=25,
        help="Storage/ambient temperature"
    )

with env_col2:
    humidity = st.slider(
        "Humidity (%)",
        min_value=40,
        max_value=95,
        value=70,
        help="Relative humidity percentage"
    )

# Prediction button
st.markdown("---")

if st.button("🔮 Predict Shelf Life", type="primary", use_container_width=True):
    # Prepare input data
    input_data = pd.DataFrame({
        "Variety": [variety],
        "Maturity_Stage": [maturity_stage],
        "Temp_C": [temperature],
        "Humidity_Pct": [humidity],
        "Packaging": [packaging],
        "Transport_KM": [transport_km]
    })
    
    # One-hot encode
    input_encoded = pd.get_dummies(input_data)
    
    # Add missing columns with 0
    for col in FEATURE_COLUMNS:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Ensure correct column order
    input_encoded = input_encoded[FEATURE_COLUMNS]
    
    # Make prediction
    prediction = model.predict(input_encoded)[0]
    
    # Display result
    st.markdown(f"""
    <div class="prediction-box">
        <p class="prediction-label">Estimated Shelf Life</p>
        <p class="prediction-value">{prediction:.1f} days</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional insights
    st.markdown("### 📊 Factors Affecting Shelf Life")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        if temperature > 28:
            st.warning("⚠️ High temperature may reduce shelf life")
        elif temperature < 20:
            st.success("✅ Cool temperature helps preserve freshness")
        
        if humidity > 80:
            st.warning("⚠️ High humidity may increase spoilage risk")
        elif humidity < 60:
            st.info("ℹ️ Low humidity may cause dehydration")
    
    with insights_col2:
        if maturity_stage >= 5:
            st.warning("⚠️ Ripe tomatoes have shorter shelf life")
        elif maturity_stage <= 2:
            st.success("✅ Less mature tomatoes last longer")
        
        if transport_km > 150:
            st.warning("⚠️ Long transport distance may affect quality")

# Footer with model info
st.markdown("---")
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    <div class="info-card">
    <strong>Model:</strong> Random Forest Regressor<br>
    <strong>Dataset:</strong> Sri Lanka Tomato Varieties<br>
    <strong>Features:</strong> Variety, Maturity Stage, Temperature, Humidity, Packaging, Transport Distance<br>
    <strong>Performance:</strong> R² Score: 0.87, MAE: 0.94 days
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Sri Lanka Tomato Varieties:**
    - **Lanka Sour**: Local variety with tangy flavor
    - **T-146**: Hybrid variety with good yield
    - **T-245**: Heat-tolerant hybrid
    - **Thilina Tomato**: Popular commercial variety
    """)
