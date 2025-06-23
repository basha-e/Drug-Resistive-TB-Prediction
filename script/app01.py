import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="DR-TB Prediction App",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        /* Import Montserrat from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Montserrat', sans-serif !important;
        }

        body {
            background-color: #f5f7fa;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .stButton>button {
            background-color: #0072C6;
            color: white;
            border-radius: 5px;
        }
        .stDownloadButton>button {
            background-color: #2ecc71;
            color: white;
        }
        .stRadio > div {
            gap: 20px;
        }
    </style>
""", unsafe_allow_html=True)


# --- Session State ---
if 'page' not in st.session_state:
    st.session_state.page = "banner"

# --- Opening Banner Page ---
def banner_page():
    st.markdown("<h1 style='text-align:center; color: #e74c3c;'>Insight-TB🧪</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:rgba(39,68,114,0.8);'>Smart prediction app for drug resistance in Tuberculosis.</h3>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h4 style='text-align:center;'>Click the button below to continue</h4>", unsafe_allow_html=True)

    # Option 1: Equal columns (most centered)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Continue"):
            st.session_state.page = "input"

    # Option 2: Wider center (your original, also fine)
    # col1, col2, col3 = st.columns([1,3,1])
    # with col2:
    #     if st.button("Continue"):
    #         st.session_state.page = "input"



# --- Input Form Page (All fields in separate rows) ---
def input_page():
    st.title("🧪 InsightTB ")
    st.markdown("### 👤 Enter patient details below to predict Drug resistance in the TB.")

    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200, value=80)
    resp_rate = st.number_input("Respiratory Rate", min_value=5, max_value=60, value=20)
    weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=60)
    culture_result = st.selectbox("MGT Sputum Culture result", ["Negative", "Positive"])
    afb_microscopy = st.selectbox("AFB Microscopy for sputum", ["Negative", "Positive"])
    tb_history = st.selectbox("History of TB disease prior enrollment", ["No", "Yes"])
    fever = st.radio("Fever", ["No", "Yes"], horizontal=True)
    weight_loss = st.radio("Weight Loss", ["No", "Yes"], horizontal=True)
    hiv_status = st.selectbox("HIV Status", ["Negative", "Positive"])

    if hiv_status == "Positive":
        cd4rslt = st.number_input("CD4 Count", min_value=0, max_value=1500, value=400)
    else:
        cd4rslt = None

    with open("model/model_rf_pipeline.pkl", "rb") as f:
        model = pickle.load(f)


    if st.button("Predict DR-TB"):
        binary_map = {"No": 0, "Yes": 1, "Negative": 0, "Positive": 1}
        gender_map = {"Female": 0, "Male": 1}
        hiv_cd4_low = 1 if hiv_status == "Positive" and cd4rslt is not None and cd4rslt < 200 else 0

        input_df = pd.DataFrame([{
            'MGT Sputum Culture result': binary_map[culture_result],
            'AFB Microscopy for sputum': binary_map[afb_microscopy],
            'Age': age,
            'Gender': gender_map[gender],
            'Heart rate': heart_rate,
            'Respiratory rate': resp_rate,
            'Weight': weight,
            'History of TB disease prior enrollment': binary_map[tb_history],
            'Fever': binary_map[fever],
            'Weight loss': binary_map[weight_loss],
            'HIV status': binary_map[hiv_status],
            'cd4rslt': cd4rslt,
            'HIV_CD4_Low': hiv_cd4_low,
        }])

        prediction = model.predict(input_df)[0]
        st.session_state.prediction_result = prediction
        st.session_state.prediction_message = "🟥 DR-TB Positive: Rifampicin Resistant" if prediction == 1 else "🟩 DR-TB Negative: Rifampicin Sensitive"

        # Store inputs in session
        st.session_state.update({
            'age': age, 'gender': gender, 'heart_rate': heart_rate,
            'resp_rate': resp_rate, 'weight': weight,
            'culture_result': culture_result, 'afb_microscopy': afb_microscopy,
            'tb_history': tb_history, 'fever': fever, 'weight_loss': weight_loss,
            'hiv_status': hiv_status, 'cd4rslt': cd4rslt
        })

        st.session_state.page = "results"

# --- Results Page (unchanged) ---
def results_page():
    st.title("📊 Prediction Results")

    if "prediction_message" in st.session_state:
        st.success(st.session_state.prediction_message)

        st.markdown("### 🧾 Patient Details")
        inputs = {
            "MGT Sputum Culture result": st.session_state.culture_result,
            "AFB Microscopy for sputum": st.session_state.afb_microscopy,
            "Age": st.session_state.age,
            "Gender": st.session_state.gender,
            "Heart Rate": st.session_state.heart_rate,
            "Respiratory Rate": st.session_state.resp_rate,
            "Weight": f"{st.session_state.weight} kg",
            "History of TB disease prior enrollment": st.session_state.tb_history,
            "Fever": st.session_state.fever,
            "Weight Loss": st.session_state.weight_loss,
            "HIV Status": st.session_state.hiv_status
        }

        if st.session_state.hiv_status == "Positive":
            inputs["CD4 Count"] = st.session_state.cd4rslt

        hiv_cd4_low = 1 if st.session_state.hiv_status == "Positive" and st.session_state.cd4rslt < 200 else 0
        inputs["HIV CD4 Low"] = hiv_cd4_low

        for k, v in inputs.items():
            st.write(f"**{k}:** {v}")

        # --- PDF GENERATION ---
        def create_pdf():
            buffer = BytesIO()
            p = canvas.Canvas(buffer, pagesize=A4)
            width, height = A4
            y = height - 50

            p.setFont("Helvetica-Bold", 20)
            p.setFillColor(colors.HexColor('#0072C6'))
            p.drawString(180, y, "InsightTB Report")
            y -= 40

            p.setFont("Helvetica", 12)
            p.setFillColor(colors.black)
            p.drawString(50, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            y -= 30

            p.line(50, y, width - 50, y)
            y -= 20

            p.setFont("Helvetica-Bold", 14)
            p.drawString(50, y, "Prediction Result:")
            y -= 20

            p.setFont("Helvetica", 12)
            if st.session_state.prediction_result == 1:
                p.setFillColorRGB(1, 0, 0)
            else:
                p.setFillColorRGB(0, 0.5, 0)
            p.drawString(70, y, st.session_state.prediction_message)
            p.setFillColor(colors.black)
            y -= 40

            p.setFont("Helvetica-Bold", 14)
            p.drawString(50, y, "Patient Details:")
            y -= 20

            p.setFont("Helvetica", 12)
            for key, val in inputs.items():
                if y < 100:
                    p.showPage()
                    y = height - 50
                p.drawString(70, y, f"{key}: {val}")
                y -= 20

            p.showPage()
            p.save()
            buffer.seek(0)
            return buffer

        pdf_file = create_pdf()
        st.download_button(
            label="📄 Download Report as PDF",
            data=pdf_file,
            file_name="report.pdf",
            mime="application/pdf"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Try Again"):
                st.session_state.page = "input"
        with col2:
            if st.button("🏠 Home"):
                st.session_state.page = "banner"
    else:
        st.error("No prediction available. Please enter patient details.")

# --- App Entry Point ---
if st.session_state.page == "banner":
    banner_page()
elif st.session_state.page == "input":
    input_page()
elif st.session_state.page == "results":
    results_page()
elif st.session_state.page == "tb_image":
    tb_image_page()

