# ============================================================
#  Heart Disease Predictor — Streamlit App
#  Champion Model: KNN (Tuned) — AUC: 89.66%, F1: 84.06%
#  Run: streamlit run app.py
#  Required files: champion_model.pkl + scaler.pkl
# ============================================================

import streamlit as st
import numpy as np
import joblib

# ── Load model & scaler ──────────────────────────────────────
model  = joblib.load('champion_model.pkl')
scaler = joblib.load('scaler.pkl')
N_FEATURES = scaler.n_features_in_

# ── Neutral hidden defaults ──────────────────────────────────
# Using healthy-class medians — NOT overall means
# to avoid biasing every prediction toward disease
HIDDEN = {
    'trestbps': 130.0,  # healthy class median BP
    'chol'    : 234.0,  # below 240 threshold = normal
    'fbs'     : 0.0,    # 85% of patients = 0
    'restecg' : 0.0,    # 0 = normal ECG
    'exang'   : 0.0,    # 0 = no exercise angina
    'slope'   : 1.0,    # 1 = flat (most common)
}

# ── Prediction function ──────────────────────────────────────
def predict(age, sex, cp, thalach, ca, oldpeak, thal):

    # Engineered features computed from user inputs
    if   age < 40: age_group = 0
    elif age < 55: age_group = 1
    elif age < 70: age_group = 2
    else:          age_group = 3

    # chol_high = 0 because hidden chol (234) < 240
    chol_high = 0

    # Features in EXACT training order (must match Phase 1)
    features = [
        age,                  # 1.  age
        sex,                  # 2.  sex
        cp,                   # 3.  cp
        HIDDEN['trestbps'],   # 4.  trestbps   ← auto
        HIDDEN['chol'],       # 5.  chol        ← auto
        HIDDEN['fbs'],        # 6.  fbs         ← auto
        HIDDEN['restecg'],    # 7.  restecg     ← auto
        thalach,              # 8.  thalach
        HIDDEN['exang'],      # 9.  exang       ← auto
        oldpeak,              # 10. oldpeak
        HIDDEN['slope'],      # 11. slope       ← auto
        ca,                   # 12. ca
        thal,                 # 13. thal
    ]

    # Add engineered features if scaler expects 15
    if N_FEATURES == 15:
        features.append(age_group)  # 14. age_group_encoded
        features.append(chol_high)  # 15. chol_high

    input_array  = np.array([features])       # shape (1, N) ✅
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
st.markdown("""
    <h2 style='text-align:center; color:#1A3C6E;'>
        ❤️ Heart Disease Risk Predictor
    </h2>
    <p style='text-align:center; color:gray;'>
        SE-CD-638 Machine Learning | KNN Champion Model (AUC: 89.66%)
    </p>
""", unsafe_allow_html=True)
st.divider()

# ── Input fields ─────────────────────────────────────────────
st.markdown("### 📋 Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "🎂 Age (years)",
        min_value=20, max_value=80, value=45, step=1
    )

    sex = st.selectbox(
        "⚧ Gender",
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
            3: "3 — No Chest Pain"
        }[x],
        help="Type of chest pain the patient experiences"
    )

    thalach = st.number_input(
        "💓 Max Heart Rate Achieved (bpm)",
        min_value=70, max_value=210, value=160, step=1,
        help="Maximum heart rate during exercise stress test"
    )

with col2:
    ca = st.selectbox(
        "🩸 Major Vessels Coloured (0–3)",
        options=[0, 1, 2, 3],
        help="Number of major vessels coloured by fluoroscopy — from angiography report"
    )

    oldpeak = st.number_input(
        "📉 ST Depression (Oldpeak)",
        min_value=0.0, max_value=6.5, value=0.0,
        step=0.1, format="%.1f",
        help="ST depression induced by exercise relative to rest — from ECG report"
    )

    thal = st.selectbox(
        "🧬 Thalassemia Result",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 — Normal",
            2: "2 — Fixed Defect",
            3: "3 — Reversible Defect"
        }[x],
        help="Thalassemia type from blood disorder test report"
    )

st.divider()

# ── Predict button ───────────────────────────────────────────
if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):
    try:
        pred, proba = predict(age, sex, cp, thalach, ca, oldpeak, thal)

        st.markdown("### 📊 Prediction Result")

        if pred == 1:
            st.error("⚠️ **Heart Disease Risk Detected**")
            st.metric("Risk Probability", f"{proba}%")
            st.progress(proba / 100)
            st.warning(
                "⚕️ This is a **screening tool only**. "
                "Please consult a cardiologist for proper diagnosis."
            )
        else:
            st.success("✅ **Low Risk — No Heart Disease Detected**")
            st.metric("Risk Probability", f"{proba}%")
            st.progress(proba / 100)
            st.info(
                "Low risk detected. Maintain a healthy lifestyle "
                "and schedule regular check-ups."
            )

        # Auto-filled values transparency
        with st.expander("ℹ️ View auto-filled background values"):
            st.caption(
                "These 6 fields were not shown on screen. "
                "They are filled with neutral healthy-class values "
                "to avoid biasing the prediction."
            )
            st.table({
                "Hidden Field"  : [
                    "Resting BP", "Cholesterol", "Fasting Sugar",
                    "Resting ECG", "Exercise Angina", "ST Slope"
                ],
                "Value Used"    : [130.0, 234.0, 0.0, 0.0, 0.0, 1.0],
                "Why Neutral"   : [
                    "Healthy class median",
                    "Below 240 = normal cholesterol",
                    "0 = no elevated sugar (85% of patients)",
                    "0 = normal ECG reading",
                    "0 = no exercise-induced angina",
                    "1 = flat slope (most common value)"
                ]
            })

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info(
            f"Scaler expects {N_FEATURES} features. "
            "Ensure champion_model.pkl and scaler.pkl are in the same folder "
            "and were saved from the same Phase 1 & 4 notebook run."
        )

# ── Test reference ───────────────────────────────────────────
with st.expander("🧪 Test: Low Risk Example Input"):
    st.markdown("""
    Use these values to verify the app gives **No Heart Disease**:
    
    | Field | Value |
    |-------|-------|
    | Age | 35 |
    | Gender | Female |
    | Chest Pain | No Chest Pain (3) |
    | Max Heart Rate | 185 |
    | Vessels (ca) | 0 |
    | ST Depression | 0.0 |
    | Thalassemia | Normal (1) |
    
    Expected result: ✅ Low Risk
    """)

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption(
    "SE-CD-638 Machine Learning | Dr. Aamir Arsalan | "
    "Heart Disease UCI Dataset | KNN Tuned — AUC 89.66% | F1 84.06%"
)
