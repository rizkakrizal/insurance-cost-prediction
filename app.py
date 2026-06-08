
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

input_user = {}

for f in FITUR:
    input_user[f] = st.number_input(
        label=f,
        value=0.0,
        step=0.1
    )

if st.button("Prediksi"):

    nilai = pd.DataFrame(
        [[input_user[f] for f in FITUR]],
        columns=FITUR
    )

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
