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
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        text-align: center;
        color: #E63946;
        font-size: 1.3rem;
        font-weight: 1000;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
        animation: fadeInDown 0.6s ease-out;
    }
    
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: 0.3px;
        animation: fadeInUp 0.6s ease-out 0.1s both;
    }
    
    .section-title {
        color: #1D3557;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #E63946;
        padding-bottom: 0.5rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #E63946 0%, #FF6B6B 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(230, 57, 70, 0.2);
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border: 2px solid rgba(255, 107, 107, 0.5);
    }
    
    .prediction-box:hover {
        box-shadow: 0 15px 50px rgba(230, 57, 70, 0.3);
        transform: translateY(-5px);
    }
    
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #FFFFFF;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.15);
        animation: scaleIn 0.5s ease-out;
    }
    
    .prediction-label {
        font-size: 1.1rem;
        color: #F1FAEE;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .info-card {
        background: linear-gradient(135deg, #F1FAEE 0%, #FFFFFF 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #E63946;
        margin: 1.2rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        border-top: 1px solid rgba(230, 57, 70, 0.1);
    }
    
    .info-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
        transform: translateX(5px);
    }
    
    .info-card strong {
        color: #1D3557;
        font-weight: 600;
    }
    
    /* Input and control styling */
    .streamlit-select {
        border: 2px solid #E8E8E8;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .streamlit-select:focus {
        border-color: #E63946;
        box-shadow: 0 0 0 3px rgba(230, 57, 70, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #E63946 0%, #D62828 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(230, 57, 70, 0.3);
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(230, 57, 70, 0.4);
        transform: translateY(-2px);
        background: linear-gradient(135deg, #D62828 0%, #B71C1C 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Horizontal divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, transparent, #E63946, transparent);
        margin: 2rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderContent {
        animation: slideDown 0.3s ease-out;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            max-height: 0;
        }
        to {
            opacity: 1;
            max-height: 1000px;
        }
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
    'Variety_Lanka Sour', 'Variety_T-146', 'Variety_T-245', 'Variety_Thilina Tomato',
    'Packaging_Plastic Crate', 'Packaging_Polysack Bag', 'Packaging_Wooden Crate'
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
    # Prepare input data with explicit one-hot encoding
    input_dict = {
        'Maturity_Stage': maturity_stage,
        'Temp_C': temperature,
        'Humidity_Pct': humidity,
        'Transport_KM': transport_km,
    }
    
    # Manually create one-hot encoded columns for Packaging
    for pkg_type in PACKAGING_TYPES:
        input_dict[f'Packaging_{pkg_type}'] = 1 if packaging == pkg_type else 0
    
    # Manually create one-hot encoded columns for Variety
    for var_type in VARIETIES:
        input_dict[f'Variety_{var_type}'] = 1 if variety == var_type else 0
    
    # Create DataFrame with all required columns in correct order
    input_encoded = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
    
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
