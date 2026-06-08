
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Prediksi Biaya Asuransi",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_artefak():
    model = joblib.load("regresi_berganda.pkl")
    scaler = joblib.load("scaler.pkl")
    fitur = joblib.load("fitur.pkl")
    return model, scaler, fitur

model, scaler, FITUR = load_artefak()

st.title("📊 Prediksi Biaya Asuransi")
st.markdown(
    "Masukkan nilai fitur lalu klik tombol Prediksi."
)

st.subheader("Input Data")

age = st.number_input("Usia", 18, 100, 25)

bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

children = st.number_input("Jumlah Anak", 0, 10, 0)

sex = st.selectbox(
    "Jenis Kelamin",
    ["Female", "Male"]
)

smoker = st.selectbox(
    "Perokok",
    ["Tidak", "Ya"]
)

region = st.selectbox(
    "Wilayah",
    ["northeast", "northwest", "southeast", "southwest"]
)

if st.button("Prediksi"):

    sex_male = 1 if sex == "Male" else 0
    smoker_yes = 1 if smoker == "Ya" else 0

    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    nilai = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": sex_male,
        "smoker_yes": smoker_yes,
        "region_northwest": region_northwest,
        "region_southeast": region_southeast,
        "region_southwest": region_southwest
    }])


    nilai_sc = scaler.transform(nilai)

    pred_log = model.predict(nilai_sc)[0]

    pred_charges = np.exp(pred_log)

    st.success(
        f"Prediksi charges_log : {pred_log:.4f}"
    )

    st.info(
        f"Estimasi biaya asuransi : {pred_charges:,.2f}"
    )

    st.subheader("Input yang Digunakan")

    st.dataframe(
        nilai,
        use_container_width=True
    )

st.divider()

st.caption(
    "Deploy Tugas 8 - Regresi Linear Berganda"
)

