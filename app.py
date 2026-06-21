import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── 1. CONFIGURATION & PAGE SETUP ──
st.set_page_config(
    page_title="Heart Disease Risk Analyzer",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Epidemiological Risk Analyzer")
st.write("Enter patient clinical metrics below to evaluate classification risk metrics.")

# ── 2. LOAD ASSETS SAVED FROM COLAB ──
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('champion_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, scaler = load_assets()

# Identify how many features the scaler expects to automatically manage dimensions
if scaler is not None:
    try:
        N_FEATURES = scaler.n_features_in_
    except AttributeError:
        # Fallback if attribute isn't present in older scikit-learn versions
        N_FEATURES = scaler.mean_.shape[0] if hasattr(scaler, 'mean_') else 13
else:
    N_FEATURES = 13


# ── 3. USER INTERFACE / INPUT FIELDS ──
st.subheader("Patient Clinical Metrics")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=1, max_value=100, value=50)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], 
                      format_func=lambda x: f"Type {x}")
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)

with col2:
    oldpeak = st.number_input("ST Depression Induced (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
    ca = st.selectbox("Major Vessels Colored by Flourosopy (ca)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3],
                        format_func=lambda x: f"Fixed/Reversible Defect ({x})" if x > 0 else f"Normal ({x})")

st.markdown("---")


# ── 4. PREDICTION EVENT TRIGGER ──
if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):
    if model is None or scaler is None:
        st.error("Model assets are missing. Please verify your repository contains champion_model.pkl and scaler.pkl.")
    else:
        # ── DEBUG BLOCK ──
        with st.expander("🔧 DEBUG — Raw input sent to model", expanded=True):
            # Formulate engineering features calculated during the pipeline stage
            age_grp = 0 if age < 40 else 1 if age < 55 else 2 if age < 70 else 3
            chol_hi = 0  # Default value assigned during preprocessing normalization
            
            # Construct base vector array mapping columns identically to training phase
            raw = [age, sex, cp, 130.0, 234.0, 0.0, 0.0,
                   thalach, 0.0, oldpeak, 1.0, ca, thal]
            
            if N_FEATURES == 15:
                raw += [age_grp, chol_hi]
            
            arr = np.array([raw])
            
            try:
                # Scale array to stop model from falling completely into the "low" class baseline bias
                scaled = scaler.transform(arr)
                
                st.write(f"**N_FEATURES scaler expects:** {N_FEATURES}")
                st.write(f"**Raw array shape input:** {arr.shape}")
                st.write(f"**Raw array contents:** {raw}")
                st.write(f"**Scaled array contents:** {scaled[0].round(3).tolist()}")
                
                # Extract neighbor metric distances from structural KNN space
                if hasattr(model, 'kneighbors'):
                    distances, indices = model.kneighbors(scaled)
                    st.write("**KNN distances to closest cluster neighbors:**")
                    st.code(f"Distances: {distances[0].round(3)}")
                else:
                    st.write("*Note: Loaded model core doesn't support direct distance evaluation natively.*")
                    
            except Exception as debug_error:
                st.error(f"Debug evaluation failed during pipeline matrix transformation: {debug_error}")

        # ── RUN PREDICTION LOGIC ──
        try:
            # Predict outcome using standard system boundaries
            prediction = model.predict(scaled)[0]
            
            # Extract probability array mapping if supported by core estimator architecture
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(scaled)[0]
                risk_probability = probabilities[1]
            else:
                risk_probability = None

            st.subheader("Model Diagnostic Result")
            
            if prediction == 1:
                st.error("🚨 **High Risk Case Classification Flagged**")
                if risk_probability is not None:
                    st.metric(label="Target Probability Confidence", value=f"{risk_probability:.2%}")
            else:
                st.success("✅ **Low Risk Case Classification Flagged**")
                if risk_probability is not None:
                    st.metric(label="Target Probability Confidence", value=f"{risk_probability:.2%}")
                    
        except Exception as pred_error:
            st.error(f"An unexpected inference anomaly occurred during processing: {pred_error}")
