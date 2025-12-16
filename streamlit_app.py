import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Analisis Clustering Kemiskinan",
    layout="wide"
)

# =========================
# TITLE
# =========================
st.title("📊 ANALISIS CLUSTERING KEMISKINAN JAWA BARAT")
st.markdown(
    "Aplikasi ini melakukan analisis dan segmentasi wilayah menggunakan metode **K-Means Clustering** "
    "berdasarkan indikator ekonomi."
)

# =========================
# LOAD DATA & MODEL
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("datasetkemiskinan_final.csv")

@st.cache_resource
def load_model():
    bundle = joblib.load("kmeans.pkl")
    return bundle["model"], bundle["scaler"]

df = load_data()
model, scaler = load_model()

features = [
    'jumlah_warga_jabar',
    'jumlah_penduduk_miskin',
    'garis_kemiskinan',   
    'jumlah_pengangguran',
    'PDRB'                      
]
X = df[features]
scaled_features = scaler.fit_transform(df[features])


st.success("✅ Dataset dan model berhasil dimuat")

# =========================
# DASHBOARD METRIC
# =========================
st.header("📈 Ringkasan Data")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Jumlah Wilayah", df['kabupaten_kota'].nunique())
with col2:
    st.metric("Total Data", len(df))
with col3:
    st.metric("Periode", f"{df['Tahun'].min()} - {df['Tahun'].max()}")

# =========================
# EDA SECTION
# =========================
st.header("🔍 Exploratory Data Analysis")

# Line Plot – Tren Penduduk Miskin
st.subheader("Tren Penduduk Miskin")

trend = df.groupby('Tahun')['jumlah_penduduk_miskin'].sum().reset_index()
fig, ax = plt.subplots()
ax.plot(trend['Tahun'], trend['jumlah_penduduk_miskin'], marker='o')
ax.set_xlabel("Tahun")
ax.set_ylabel("Penduduk Miskin")
ax.grid(True)
st.pyplot(fig)

# Scatter Plot – PDRB vs Kemiskinan
st.subheader("Hubungan PDRB dan Kemiskinan")

fig, ax = plt.subplots()
ax.scatter(df['PDRB'], df['jumlah_penduduk_miskin'])
ax.set_xlabel("PDRB")
ax.set_ylabel("Penduduk Miskin")
ax.grid(True)
st.pyplot(fig)

# =========================
# CLUSTER ANALYSIS
# =========================
st.header("🎯 Hasil Clustering")

for i in sorted(df['Cluster'].unique()):
    data_c = df[df['Cluster'] == i]

    kategori = (
        "Kemiskinan Tinggi"
        if data_c['jumlah_penduduk_miskin'].mean() > df['jumlah_penduduk_miskin'].mean()
        else "Kemiskinan Rendah"
    )

    with st.expander(f"🔵 Cluster {i} – {kategori}"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Indikator Ekonomi (Rata-rata):**")
            st.write(f"- PDRB: Rp {data_c['PDRB'].mean():,.0f}")
            st.write(f"- Garis Kemiskinan: Rp {data_c['garis_kemiskinan'].mean():,.0f}")

        with col2:
            st.write("**Kondisi Sosial:**")
            st.write(f"- Penduduk Miskin: {data_c['jumlah_penduduk_miskin'].mean():.1f} ribu")
            st.write(f"- Pengangguran: {data_c['jumlah_pengangguran'].mean():,.0f} jiwa")

        st.write("**Contoh Wilayah:**")
        for w in data_c['kabupaten_kota'].unique()[:5]:
            st.write(f"- {w}")


# =========================
# DATA TABLE
# =========================
st.header("📋 Data Lengkap")
st.dataframe(df, use_container_width=True)

# =========================
# FOOTER
# =========================
st.divider()
st.caption("Proyek Akhir Machine Learning | Teknik Informatika | 2025")










