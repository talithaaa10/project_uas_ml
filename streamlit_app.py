import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# ==================== CONFIG ====================
st.set_page_config(
    page_title="Analisis Kemiskinan Jawa Barat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== NAMA CLUSTER ====================
CLUSTER_NAMES = {
    "0": "Cluster 0 (Kemiskinan Rendah / Relatif Sejahtera)",
    "1": "Cluster 1 (Kemiskinan Sedang)",
    "2": "Cluster 2 (Kemiskinan Tinggi / Perlu Perhatian)",
    "3": "Cluster 3"
}

# ==================== CSS ====================
def local_css():
    st.markdown("""
    <style>
        div[data-testid="stMetric"] {
            background-color: #F0F2F6;
            border: 1px solid #D6D6D6;
            padding: 15px;
            border-radius: 10px;
        }
        .main-title {
            font-size: 2.4em;
            color: #2E86C1;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .cluster-card {
            background-color: #e8f4f8;
            padding: 18px;
            border-radius: 10px;
            margin-bottom: 12px;
            border-left: 5px solid #2E86C1;
        }
    </style>
    """, unsafe_allow_html=True)

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    if not os.path.exists("datasetkemiskinan_final.csv") or not os.path.exists("dataset_kemiskinan.csv"):
        st.error("File dataset tidak ditemukan. Pastikan CSV ada di root project.")
        st.stop()

    df = pd.read_csv("datasetkemiskinan_final.csv")
    df_raw = pd.read_csv("dataset_kemiskinan.csv")

    if "Tahun" in df.columns:
        df["Tahun"] = df["Tahun"].astype(int)

    return df, df_raw

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    if not os.path.exists("kmeans.pkl"):
        st.error("File model kmeans.pkl tidak ditemukan.")
        st.stop()

    bundle = joblib.load("kmeans.pkl")

    if "model" not in bundle or "scaler" not in bundle:
        st.error("Isi kmeans.pkl tidak valid. Harus berisi 'model' dan 'scaler'.")
        st.stop()

    return bundle["model"], bundle["scaler"]

# ==================== CLUSTERING ====================
def apply_clustering(df, model, scaler):
    fitur = [
        "jumlah_warga_jabar",
        "jumlah_penduduk_miskin",
        "garis_kemiskinan",
        "jumlah_pengangguran",
        "PDRB"
    ]

    missing = [f for f in fitur if f not in df.columns]
    if missing:
        st.error(f"Kolom berikut tidak ditemukan di dataset: {missing}")
        st.stop()

    X = df[fitur]
    X_scaled = scaler.transform(X)
    df["Cluster"] = model.predict(X_scaled).astype(str)
    df["Cluster Label"] = df["Cluster"].map(CLUSTER_NAMES).fillna(df["Cluster"])

    return df

# ==================== VISUAL ====================
def plot_cluster_scatter(df):
    fig = px.scatter(
        df,
        x="PDRB",
        y="jumlah_penduduk_miskin",
        color="Cluster Label",
        size="jumlah_pengangguran",
        hover_name="kabupaten_kota",
        title="Peta Sebaran PDRB vs Penduduk Miskin",
        template="plotly_white",
        height=520
    )
    return fig

# ==================== MAIN ====================
def main():
    local_css()
    df, df_raw = load_data()
    model, scaler = load_model()
    df = apply_clustering(df, model, scaler)

    # ===== SIDEBAR =====
    with st.sidebar:
        st.title("📊 Jabar Analytics")
        st.info("Aplikasi Clustering K-Means Kemiskinan Jawa Barat")
        menu = st.radio("Navigasi", ["🏠 Dashboard", "📈 EDA", "🎯 Hasil Clustering", "📂 Dataset"])

        st.divider()
        st.markdown("**Keterangan Cluster:**")
        for k, v in CLUSTER_NAMES.items():
            if k in df["Cluster"].unique():
                st.caption(f"**{k}** : {v}")

    # ===== DASHBOARD =====
    if menu == "🏠 Dashboard":
        st.markdown('<div class="main-title">Dashboard Kemiskinan Jawa Barat</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah Wilayah", df["kabupaten_kota"].nunique())
        c2.metric("Rata-rata PDRB", f"Rp {df['PDRB'].mean():,.0f}")
        c3.metric("Total Penduduk Miskin", f"{df['jumlah_penduduk_miskin'].sum():,.0f}")
        c4.metric("Jumlah Cluster", df["Cluster"].nunique())

        st.divider()

        cluster_count = df["Cluster Label"].value_counts().reset_index()
        cluster_count.columns = ["Cluster", "Jumlah"]

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                px.bar(cluster_count, x="Jumlah", y="Cluster", orientation="h",
                       title="Jumlah Wilayah per Cluster"),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                px.pie(cluster_count, values="Jumlah", names="Cluster", hole=0.4,
                       title="Proporsi Cluster"),
                use_container_width=True
            )

    # ===== EDA =====
    elif menu == "📈 EDA":
        st.markdown("<h2>Exploratory Data Analysis</h2>", unsafe_allow_html=True)

        yearly = df_raw.groupby("Tahun").sum(numeric_only=True).reset_index()
        fig = px.line(yearly, x="Tahun", y="jumlah_penduduk_miskin",
                      title="Tren Total Penduduk Miskin")
        st.plotly_chart(fig, use_container_width=True)

    # ===== CLUSTER DETAIL =====
    elif menu == "🎯 Hasil Clustering":
        st.markdown("<h2>Analisis Hasil Clustering</h2>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Visualisasi", "Detail Cluster"])

        with tab1:
            st.plotly_chart(plot_cluster_scatter(df), use_container_width=True)

        with tab2:
            for c in sorted(df["Cluster"].unique()):
                c_data = df[df["Cluster"] == c]
                c_name = CLUSTER_NAMES.get(c, f"Cluster {c}")

                st.markdown(f"""
                <div class="cluster-card">
                    <h3>{c_name}</h3>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("Rata-rata PDRB", f"Rp {c_data['PDRB'].mean():,.0f}")
                col2.metric("Rata-rata Penduduk Miskin", f"{c_data['jumlah_penduduk_miskin'].mean():.1f}")
                col3.metric("Rata-rata Pengangguran", f"{c_data['jumlah_pengangguran'].mean():,.0f}")

                with st.expander("Lihat contoh wilayah"):
                    st.table(c_data[["kabupaten_kota", "Tahun", "jumlah_penduduk_miskin"]].head(10))

    # ===== DATASET =====
    else:
        st.markdown("<h2>Dataset & Filter</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            selected_cluster = st.multiselect(
                "Pilih Cluster",
                sorted(df["Cluster"].unique()),
                default=sorted(df["Cluster"].unique())
            )

        with col2:
            if "Tahun" in df.columns:
                selected_year = st.multiselect(
                    "Pilih Tahun",
                    sorted(df["Tahun"].unique()),
                    default=sorted(df["Tahun"].unique())
                )
                filtered = df[
                    (df["Cluster"].isin(selected_cluster)) &
                    (df["Tahun"].isin(selected_year))
                ]
            else:
                filtered = df[df["Cluster"].isin(selected_cluster)]

        st.dataframe(filtered, use_container_width=True)

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            "hasil_clustering_kemiskinan.csv",
            "text/csv"
        )

# ==================== RUN ====================
if __name__ == "__main__":
    main()
