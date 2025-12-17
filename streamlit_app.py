import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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
    """Load dataset"""
    df = pd.read_csv('datasetkemiskinan_final.csv')
    df_raw = pd.read_csv('dataset_kemiskinan.csv')
    return df, df_raw

@st.cache_resource
def load_model():
    """Load KMeans model"""
    bundle = joblib.load("kmeans.pkl")
    return bundle["model"], bundle["scaler"]

def apply_clustering(df, model, scaler):
    fitur = [
        'jumlah_warga_jabar',
        'jumlah_penduduk_miskin',
        'garis_kemiskinan',
        'jumlah_pengangguran',
        'PDRB'
    ]

    X = df[fitur]
    X_scaled = scaler.transform(X)
    df["Cluster"] = model.predict(X_scaled)
    return df

def build_cluster_data(df):
    """Mengganti isi hasil_clustering.json"""
    cluster_data = {
        "skor_silhouette": 0.0,
        "skor_davies_bouldin": 0.0,
        "cluster": {}
    }

    for i in sorted(df["Cluster"].unique()):
        cdf = df[df["Cluster"] == i]

        cluster_data["cluster"][str(i)] = {
            "kategori": f"Cluster {i}",
            "jumlah": len(cdf),
            "pdrb_rata": cdf["PDRB"].mean(),
            "miskin_rata": cdf["jumlah_penduduk_miskin"].mean(),
            "garis_kemiskinan_rata": cdf["garis_kemiskinan"].mean(),
            "pengangguran_rata": cdf["jumlah_pengangguran"].mean(),
            "contoh_wilayah": cdf["kabupaten_kota"].unique()[:5].tolist()
        }

    return cluster_data

# ==================== VISUAL FUNCTIONS (TETAP) ====================
def create_line_plot(df_raw):
    annual_stats = df_raw.groupby('Tahun').agg({
        'jumlah_penduduk_miskin': 'sum',
        'garis_kemiskinan': 'mean'
    }).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(annual_stats['Tahun'], annual_stats['jumlah_penduduk_miskin'], marker='o')
    ax1.set_title('Tren Total Penduduk Miskin')

    ax2.plot(annual_stats['Tahun'], annual_stats['garis_kemiskinan'], marker='o')
    ax2.set_title('Tren Garis Kemiskinan')

    plt.tight_layout()
    return fig

def create_box_plots(df_raw):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=df_raw, x='Tahun', y='jumlah_penduduk_miskin', ax=ax1)
    sns.boxplot(data=df_raw, x='Tahun', y='jumlah_pengangguran', ax=ax2)
    plt.tight_layout()
    return fig

def create_scatter_plot(df_raw):
    df_2019 = df_raw[df_raw['Tahun'] == 2019]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_2019['PDRB'], df_2019['jumlah_penduduk_miskin'], alpha=0.6)
    ax.set_title("PDRB vs Penduduk Miskin (2019)")
    return fig

def create_correlation_matrix(df_raw):
    df_2019 = df_raw[df_raw['Tahun'] == 2019]
    corr = df_2019[['jumlah_warga_jabar','jumlah_penduduk_miskin',
                    'garis_kemiskinan','jumlah_pengangguran','PDRB']].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    return fig

def create_cluster_comparison(cluster_data):
    clusters = list(cluster_data["cluster"].keys())
    pdrb = [cluster_data["cluster"][c]["pdrb_rata"] for c in clusters]
    miskin = [cluster_data["cluster"][c]["miskin_rata"] for c in clusters]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(clusters, pdrb)
    ax1.set_title("PDRB Rata-rata")

    ax2.bar(clusters, miskin)
    ax2.set_title("Penduduk Miskin Rata-rata")

    plt.tight_layout()
    return fig

# ==================== MAIN APP ====================
def main():
    df, df_raw = load_data()
    model, scaler = load_model()
    df = apply_clustering(df, model, scaler)
    cluster_data = build_cluster_data(df)

    # ===== SIDEBAR =====
    with st.sidebar:
        st.title("📊 Menu Analisis")
        menu = st.radio(
            "Pilih Analisis:",
            ["🏠 Dashboard", "📈 EDA & Visualisasi", "🎯 Hasil Clustering",
             "📋 Dataset", "💡 Insight & Rekomendasi"]
        )

    # ===== DASHBOARD =====
    if menu == "🏠 Dashboard":
        st.title("📊 DASHBOARD ANALISIS CLUSTERING")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Data", len(df))
        col2.metric("Jumlah Wilayah", df['kabupaten_kota'].nunique())
        col3.metric("Periode Data", f"{df['Tahun'].min()}-{df['Tahun'].max()}")
        col4.metric("Jumlah Cluster", df['Cluster'].nunique())

        st.subheader("📊 Visualisasi Cepat")
        st.pyplot(create_cluster_comparison(cluster_data))

    elif menu == "📈 EDA & Visualisasi":
        st.pyplot(create_line_plot(df_raw))
        st.pyplot(create_box_plots(df_raw))
        st.pyplot(create_scatter_plot(df_raw))
        st.pyplot(create_correlation_matrix(df_raw))

    elif menu == "🎯 Hasil Clustering":
        for i in cluster_data["cluster"]:
            c = cluster_data["cluster"][i]
            st.subheader(f"Cluster {i}")
            st.write(c)

    elif menu == "📋 Dataset":
        st.dataframe(df, use_container_width=True)

    else:
        st.info("Insight & rekomendasi tetap sama seperti versi JSON")

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
