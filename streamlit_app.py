import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ==================== CONFIG ====================
st.set_page_config(
    page_title="Analisis Kemiskinan Jawa Barat",
    page_icon="📊",
    layout="wide"
)

# ==================== CLUSTER LABEL ====================
CLUSTER_NAMES = {
    "0": "Cluster 0 (Kemiskinan Rendah)",
    "1": "Cluster 1 (Kemiskinan Sedang)",
    "2": "Cluster 2 (Kemiskinan Tinggi)"
}

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    if not os.path.exists("datasetkemiskinan_final.csv"):
        st.error("datasetkemiskinan_final.csv tidak ditemukan")
        st.stop()

    if not os.path.exists("dataset_kemiskinan.csv"):
        st.error("dataset_kemiskinan.csv tidak ditemukan")
        st.stop()

    df = pd.read_csv("datasetkemiskinan_final.csv")
    df_raw = pd.read_csv("dataset_kemiskinan.csv")

    df["Tahun"] = df["Tahun"].astype(int)

    return df, df_raw

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    if not os.path.exists("kmeans.pkl"):
        st.error("File kmeans.pkl tidak ditemukan")
        st.stop()

    bundle = joblib.load("kmeans.pkl")

    return bundle["model"], bundle["scaler"]

# ==================== APPLY CLUSTER ====================
def apply_clustering(df, model, scaler):
    fitur = [
        "jumlah_warga_jabar",
        "jumlah_penduduk_miskin",
        "garis_kemiskinan",
        "jumlah_pengangguran",
        "PDRB"
    ]

    X = df[fitur]
    X_scaled = scaler.transform(X)

    df["Cluster"] = model.predict(X_scaled).astype(str)
    df["Cluster Label"] = df["Cluster"].map(CLUSTER_NAMES)

    return df

# ==================== VISUAL ====================
def scatter_plot(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="PDRB",
        y="jumlah_penduduk_miskin",
        hue="Cluster Label",
        size="jumlah_pengangguran",
        ax=ax
    )
    ax.set_title("PDRB vs Penduduk Miskin")
    ax.grid(True, alpha=0.3)
    return fig

# ==================== MAIN ====================
def main():
    df, df_raw = load_data()
    model, scaler = load_model()
    df = apply_clustering(df, model, scaler)

    # ===== SIDEBAR =====
    with st.sidebar:
        st.title("📊 Menu")
        menu = st.radio(
            "Pilih Menu",
            ["🏠 Dashboard", "📈 EDA", "🎯 Clustering", "📋 Dataset"]
        )

    # ===== DASHBOARD =====
    if menu == "🏠 Dashboard":
        st.title("Dashboard Analisis Kemiskinan Jawa Barat")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Jumlah Wilayah", df["kabupaten_kota"].nunique())
        col2.metric("Rata-rata PDRB", f"Rp {df['PDRB'].mean():,.0f}")
        col3.metric("Total Penduduk Miskin", f"{df['jumlah_penduduk_miskin'].sum():,.0f}")
        col4.metric("Jumlah Cluster", df["Cluster"].nunique())

        st.divider()

        cluster_count = df["Cluster Label"].value_counts()

        fig, ax = plt.subplots()
        cluster_count.plot(kind="bar", ax=ax)
        ax.set_title("Jumlah Wilayah per Cluster")
        st.pyplot(fig)

    # ===== EDA =====
    elif menu == "📈 EDA":
        st.title("Exploratory Data Analysis")

        yearly = df_raw.groupby("Tahun")["jumlah_penduduk_miskin"].sum()

        fig, ax = plt.subplots()
        ax.plot(yearly.index, yearly.values, marker="o")
        ax.set_title("Tren Penduduk Miskin")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Jumlah Penduduk Miskin")
        ax.grid(True)

        st.pyplot(fig)

    # ===== CLUSTERING =====
    elif menu == "🎯 Clustering":
        st.title("Hasil Clustering")

        st.pyplot(scatter_plot(df))

        for c in sorted(df["Cluster"].unique()):
            c_data = df[df["Cluster"] == c]
            st.subheader(CLUSTER_NAMES[c])

            col1, col2, col3 = st.columns(3)
            col1.metric("Rata-rata PDRB", f"Rp {c_data['PDRB'].mean():,.0f}")
            col2.metric("Penduduk Miskin", f"{c_data['jumlah_penduduk_miskin'].mean():.1f}")
            col3.metric("Pengangguran", f"{c_data['jumlah_pengangguran'].mean():,.0f}")

    # ===== DATASET =====
    else:
        st.title("Dataset")

        cluster_filter = st.multiselect(
            "Pilih Cluster",
            sorted(df["Cluster"].unique()),
            default=sorted(df["Cluster"].unique())
        )

        filtered = df[df["Cluster"].isin(cluster_filter)]
        st.dataframe(filtered, use_container_width=True)

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            "hasil_clustering.csv",
            "text/csv"
        )

# ==================== RUN ====================
if __name__ == "__main__":
    main()
