import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ==================== SETUP ====================
st.set_page_config(
    page_title="Analisis Clustering Kemiskinan Jabar",
    page_icon="📊",
    layout="wide"
)

# ==================== FUNCTIONS ====================
@st.cache_data
def load_data():
    # ==============================
    # LOAD MODEL
    # ==============================
    model_bundle = joblib.load("model.pkl")
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]

    silhouette = model_bundle["silhouette"]
    davies = model_bundle["davies_bouldin"]

    # ==============================
    # LOAD DATASET
    # ==============================
    df = pd.read_csv("datasetkemiskinan_final.csv")
    df_raw = pd.read_csv("dataset_kemiskinan.csv")

    # ==============================
    # PREDICT CLUSTER
    # ==============================
    fitur = [
        "jumlah_warga_jabar",
        "jumlah_penduduk_miskin",
        "garis_kemiskinan",
        "jumlah_pengangguran",
        "PDRB"
    ]

    X = scaler.transform(df[fitur])
    df["Cluster"] = model.predict(X)

    # ==============================
    # BUILD cluster_data (JSON-LIKE)
    # ==============================
    cluster_data = {
        "skor_silhouette": silhouette,
        "skor_davies_bouldin": davies,
        "cluster": {}
    }

    for i in sorted(df["Cluster"].unique()):
        data_cluster = df[df["Cluster"] == i]

        cluster_data["cluster"][str(i)] = {
            "kategori": f"Cluster {i}",
            "jumlah": len(data_cluster),
            "pdrb_rata": data_cluster["PDRB"].mean(),
            "miskin_rata": data_cluster["jumlah_penduduk_miskin"].mean(),
            "garis_kemiskinan_rata": data_cluster["garis_kemiskinan"].mean(),
            "pengangguran_rata": data_cluster["jumlah_pengangguran"].mean(),
            "contoh_wilayah": data_cluster["kabupaten_kota"].head(5).tolist()
        }

    return cluster_data, df, df_raw


def create_line_plot(df_raw):
    annual_stats = df_raw.groupby('Tahun').agg({
        'jumlah_penduduk_miskin': 'sum',
        'garis_kemiskinan': 'mean'
    }).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(
        annual_stats['Tahun'],
        annual_stats['jumlah_penduduk_miskin'],
        marker='o',
        linewidth=2.5
    )
    ax1.set_title('Tren Total Penduduk Miskin')
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        annual_stats['Tahun'],
        annual_stats['garis_kemiskinan'],
        marker='o',
        linewidth=2.5
    )
    ax2.set_title('Tren Rata-rata Garis Kemiskinan')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_box_plots(df_raw):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    sns.boxplot(data=df_raw, x='Tahun', y='jumlah_penduduk_miskin', ax=ax1)
    ax1.set_title('Distribusi Penduduk Miskin')

    sns.boxplot(data=df_raw, x='Tahun', y='jumlah_pengangguran', ax=ax2)
    ax2.set_title('Distribusi Pengangguran')

    plt.tight_layout()
    return fig


def create_scatter_plot(df_raw):
    df_2019 = df_raw[df_raw['Tahun'] == 2019]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        df_2019['PDRB'],
        df_2019['jumlah_penduduk_miskin'],
        s=df_2019['jumlah_warga_jabar'] / 10,
        alpha=0.6
    )
    ax.set_title('PDRB vs Penduduk Miskin (2019)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_correlation_matrix(df_raw):
    df_2019 = df_raw[df_raw['Tahun'] == 2019]

    cols = [
        'jumlah_warga_jabar',
        'jumlah_penduduk_miskin',
        'garis_kemiskinan',
        'jumlah_pengangguran',
        'PDRB'
    ]

    corr = df_2019[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", linewidths=1, ax=ax)
    ax.set_title('Matriks Korelasi (2019)')

    plt.tight_layout()
    return fig


def create_cluster_comparison(cluster_data):
    clusters = ['Cluster 0', 'Cluster 1', 'Cluster 2']
    pdrb = [cluster_data['cluster'][str(i)]['pdrb_rata'] for i in range(3)]
    miskin = [cluster_data['cluster'][str(i)]['miskin_rata'] for i in range(3)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(clusters, pdrb)
    ax1.set_title("PDRB Rata-rata per Cluster")

    ax2.bar(clusters, miskin)
    ax2.set_title("Penduduk Miskin Rata-rata per Cluster")

    plt.tight_layout()
    return fig


# ==================== MAIN APP ====================
def main():
    cluster_data, df, df_raw = load_data()

    with st.sidebar:
        st.title("📊 Menu Analisis")
        menu = st.radio(
            "Pilih Analisis:",
            [
                "🏠 Dashboard",
                "📈 EDA & Visualisasi",
                "🎯 Hasil Clustering",
                "📋 Dataset",
                "💡 Insight & Rekomendasi"
            ]
        )

        st.divider()
        st.metric("Silhouette Score", f"{cluster_data['skor_silhouette']:.3f}")
        st.metric("Davies-Bouldin", f"{cluster_data['skor_davies_bouldin']:.3f}")

    if menu == "🏠 Dashboard":
        st.title("📊 Dashboard Analisis")
        st.pyplot(create_cluster_comparison(cluster_data))

    elif menu == "📈 EDA & Visualisasi":
        st.title("📈 Exploratory Data Analysis")
        st.pyplot(create_line_plot(df_raw))
        st.pyplot(create_box_plots(df_raw))
        st.pyplot(create_scatter_plot(df_raw))
        st.pyplot(create_correlation_matrix(df_raw))

    elif menu == "🎯 Hasil Clustering":
        st.title("🎯 Hasil Clustering")
        for i in range(3):
            c = cluster_data["cluster"][str(i)]
            st.subheader(f"Cluster {i}")
            st.write(c)

    elif menu == "📋 Dataset":
        st.dataframe(df, use_container_width=True)

    else:
        st.title("💡 Insight & Rekomendasi")
        st.info("Rekomendasi kebijakan berbasis hasil clustering")

# ==================== RUN ====================
if __name__ == "__main__":
    main()
