# ============================================================
#  Heart Disease Predictor — Streamlit App (FINAL REMIXED UI)
#  Champion: KNN (Tuned) — AUC: 89.66%, F1: 84.06%
#  Run: streamlit run app.py
#  Required: champion_model.pkl + scaler.pkl
# ============================================================

import streamlit as st
import numpy as np
import joblib

# ── Load model & scaler ──────────────────────────────────────
@st.cache_resource
def load_assets():
    model = joblib.load('champion_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    N_FEATURES = scaler.n_features_in_
except Exception as e:
    st.error(f"Initialization Error: Could not load model or scaler assets. {e}")
    st.stop()

# ── Hidden field defaults ────────────────────────────────────
HIDDEN = {
    'trestbps': 130.0,   # healthy median
    'chol'    : 212.0,   # healthy patient typical value
    'fbs'     : 0.0,     # 0 = normal fasting sugar
    'restecg' : 1.0,     # 1 = most common in healthy patients
    'exang'   : 0.0,     # 0 = no exercise angina
    'slope'   : 2.0,     # 2 = most common in healthy patients
}

# ── Prediction function ──────────────────────────────────────
def predict(age, sex, cp, thalach, ca, oldpeak, thal):
    # age_group from user age
    if   age < 40: age_group = 0
    elif age < 55: age_group = 1
    elif age < 70: age_group = 2
    else:          age_group = 3

    # chol_high from hidden chol (212 < 240 = 0 = normal)
    chol_high = 1 if HIDDEN['chol'] > 240 else 0  # always 0

    # Features in EXACT column order from X_train
    features = [
        age,                  # 1.  age
        sex,                  # 2.  sex
        cp,                   # 3.  cp
        HIDDEN['trestbps'],   # 4.  trestbps
        HIDDEN['chol'],       # 5.  chol
        HIDDEN['fbs'],        # 6.  fbs
        HIDDEN['restecg'],    # 7.  restecg
        thalach,              # 8.  thalach
        HIDDEN['exang'],      # 9.  exang
        oldpeak,              # 10. oldpeak
        HIDDEN['slope'],      # 11. slope
        ca,                   # 12. ca
        thal,                 # 13. thal
    ]

    if N_FEATURES == 15:
        features.append(age_group)   # 14. age_group_encoded
        features.append(chol_high)   # 15. chol_high

    input_array  = np.array([features])
    input_scaled = scaler.transform(input_array)
    pred  = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    return int(pred), round(float(proba) * 100, 1)


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# ── Header ───────────────────────────────────────────────────
st.title("❤️ Heart Disease Risk Predictor")
st.caption("⚡ Model Engine: Tuned K-Nearest Neighbors (KNN) | AUC: 89.66% | F1: 84.06%")
st.write("---")

# ── Input fields ─────────────────────────────────────────────
st.subheader("📋 Patient Diagnostic Parameters")

# Wrapping inputs inside a clean UI container/card block
with st.container(border=True):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        age = st.number_input(
            "🎂 Age (years)",
            min_value=20, max_value=80, value=52, step=1
        )

        sex = st.selectbox(
            "⚧ Biological Sex",
            options=[1, 0],
            format_func=lambda x: "Male" if x == 1 else "Female"
        )

        cp = st.selectbox(
            "💔 Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 — Typical Angina",
                1: "1 — Atypical Angina",
                2: "2 — Non-Anginal Pain",
                3: "3 — Asymptomatic"
            }[x],
            help="Clinical classification of chest pain expression."
        )

        thalach = st.number_input(
            "💓 Max Heart Rate Achieved (bpm)",
            min_value=70, max_value=210, value=168, step=1,
            help="Maximum heart rate achieved during exercise stress test protocols."
        )

    with col2:
        ca = st.selectbox(
            "🩸 Major Vessels Colored (0–3)",
            options=[0, 1, 2, 3],
            index=2,   # default = 2 (matches healthy patient profile)
            help="Number of major vessels structuralized by fluoroscopy assessment."
        )

        oldpeak = st.number_input(
            "📉 ST Depression (Oldpeak)",
            min_value=0.0, max_value=6.5, value=1.0,
            step=0.1, format="%.1f",
            help="ST segment depression induced by exercise relative to rest states."
        )

        thal = st.selectbox(
            "🧬 Thalassemia Assessment",
            options=[1, 2, 3],
            index=2,   # default = 3 (most common in healthy patients)
            format_func=lambda x: {
                1: "1 — Normal",
                2: "2 — Fixed Defect",
                3: "3 — Reversible Defect"
            }[x],
            help="Blood-disorder mapping from specialized medical panels."
        )

st.write("")

# ── Predict button ───────────────────────────────────────────
if st.button("🔍 Execute Risk Calculation", use_container_width=True, type="primary"):
    try:
        pred, proba = predict(age, sex, cp, thalach, ca, oldpeak, thal)

        st.write("")
        st.subheader("📊 Analytical Outcomes")
        
        # Output card block
        with st.container(border=True):
            metric_col, progress_col = st.columns([1, 2], vertical_alignment="center")
            
            with metric_col:
                st.metric(label="Calculated Risk Probability", value=f"{proba}%")
                
            with progress_col:
                # Changes color dynamically depending on threat level
                if pred == 1:
                    st.error("⚠️ **High Risk of Heart Disease Detected**")
                else:
                    st.success("✅ **Low Risk — No Sign of Disease Detected**")
                    
                st.progress(proba / 100)

            # Warning / Info Message Disclaimers
            if pred == 1:
                st.warning(
                    "⚕️ **Clinical Screening Disclaimer:** This calculation is an AI-driven screening probability evaluation. "
                    "It must not supersede definitive evaluation by a licensed cardiologist."
                )
            else:
                st.info(
                    "💡 **Patient Status Note:** Low statistical correlation detected based on parameters provided. "
                    "Continue managing optimal dietary habits and standard diagnostic follow-ups."
                )

        # Show auto-filled values
        with st.expander("ℹ️ View Imputed Background Metrics"):
            st.caption(
                "These 6 features were automatically processed in the background using statistical averages "
                "of baseline healthy profiles built directly from training historical data."
            )
            st.table({
                "Hidden Metric Field": [
                    "Resting Blood Pressure", "Total Serum Cholesterol", "Fasting Blood Sugar",
                    "Resting ECG Outcome", "Exercise Induced Angina", "Peak Exercise ST Slope"
                ],
                "Value Applied": [130.0, 212.0, 0.0, 1.0, 0.0, 2.0],
                "Statistical Basis": [
                    "Healthy population median",
                    "Normal physiological range boundary (< 240)",
                    "0 = Normal (< 120 mg/dl; 85% representation)",
                    "1 = Baseline standard for asymptomatic individuals",
                    "0 = Negative for exercise induced ischemia",
                    "2 = Typical performance profile for healthy subjects"
                ]
            })

    except Exception as e:
        st.error(f"❌ Diagnostic Pipeline Failure: {str(e)}")
        st.info(
            f"Expected Matrix dimension configuration: {N_FEATURES} features. "
            "Verify environment contains valid deployment footprints for 'champion_model.pkl' and 'scaler.pkl'."
        )

# ── Verified test case ───────────────────────────────────────
with st.expander("🧪 Access Integration Test Case"):
    st.markdown("""
    Provide these baseline matrices obtained from historical records of an asymptomatic healthy patient to verify system sanity:
    """)
    
    test_data = {
        "Parameter Feature Label": [
            "Age", "Biological Sex", "Chest Pain Classification", 
            "Max Heart Rate Achieved", "Major Vessels (ca)", 
            "ST Depression (Oldpeak)", "Thalassemia Result"
        ],
        "Input Value Target": [
            "52", "Male (1)", "0 — Typical Angina", 
            "168", "2", "1.0", "3 — Reversible Defect"
        ]
    }
    st.table(test_data)
    st.info("🎯 **Expected Standard Output Resolution:** Low Risk Profile (KNN Predicted Value: 0% Probability Risk)")

# ── Footer ───────────────────────────────────────────────────
st.write("---")
st.caption("🩺 Processing Node: UCI Heart Disease Pipeline Engine | KNN Core Engine Validation Target v1.0.2")
