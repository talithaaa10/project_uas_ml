import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==================== CONFIG ====================
st.set_page_config(
    page_title="Analisis Clustering Kemiskinan Jabar",
    page_icon="📊",
    layout="wide"
)

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    df = pd.read_csv("datasetkemiskinan_final.csv")
    df_raw = pd.read_csv("dataset_kemiskinan.csv")
    return df, df_raw

@st.cache_resource
def load_model():
    model = joblib.load("kmeans.pkl")
    return model

# ==================== KETERANGAN CLUSTER ====================
CLUSTER_LABEL = {
    0: "Kemiskinan Rendah",
    1: "Kemiskinan Sedang",
    2: "Kemiskinan Tinggi"
}

# ==================== CLUSTERING ====================
def apply_clustering(df, model):
    fitur = [
        "jumlah_warga_jabar",
        "jumlah_penduduk_miskin",
        "garis_kemiskinan",
        "jumlah_pengangguran",
        "PDRB"
    ]

    X = df[fitur]
    df["Cluster"] = model.predict(X)
    return df

# ==================== EDA ====================
def line_plot(df_raw):
    yearly = df_raw.groupby("Tahun").agg({
        "jumlah_penduduk_miskin": "sum",
        "garis_kemiskinan": "mean"
    }).reset_index()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(yearly["Tahun"], yearly["jumlah_penduduk_miskin"], marker="o")
    ax[0].set_title("Tren Penduduk Miskin")

    ax[1].plot(yearly["Tahun"], yearly["garis_kemiskinan"], marker="o")
    ax[1].set_title("Tren Garis Kemiskinan")

    plt.tight_layout()
    return fig


def box_plot(df_raw):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_raw, x="Tahun", y="jumlah_penduduk_miskin", ax=ax)
    ax.set_title("Distribusi Penduduk Miskin")
    return fig


def correlation_plot(df_raw):
    df_2019 = df_raw[df_raw["Tahun"] == 2019]
    corr = df_2019.select_dtypes(include=np.number).corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Matriks Korelasi (2019)")
    return fig

# ==================== MAIN ====================
def main():
    df, df_raw = load_data()
    model = load_model()
    df = apply_clustering(df, model)

    st.sidebar.title("📊 Menu")
    menu = st.sidebar.radio(
        "Pilih Halaman",
        ["Dashboard", "EDA", "Hasil Clustering", "Dataset"]
    )

    # ========== DASHBOARD ==========
    if menu == "Dashboard":
        st.title("📊 Dashboard Clustering Kemiskinan Jawa Barat")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Data", len(df))
        col2.metric("Jumlah Wilayah", df["kabupaten_kota"].nunique())
        col3.metric("Jumlah Cluster", df["Cluster"].nunique())

        st.divider()
        st.subheader("Distribusi Cluster")
        st.bar_chart(df["Cluster"].value_counts())

        st.subheader("Keterangan Cluster")
        for k, v in CLUSTER_LABEL.items():
            st.write(f"Cluster {k} : {v}")

    # ========== EDA ==========
    elif menu == "EDA":
        st.title("📈 Exploratory Data Analysis")
        st.pyplot(line_plot(df_raw))
        st.pyplot(box_plot(df_raw))
        st.pyplot(correlation_plot(df_raw))

    # ========== HASIL CLUSTERING ==========
    elif menu == "Hasil Clustering":
        st.title("🎯 Hasil Clustering")

        for i in sorted(df["Cluster"].unique()):
            cluster_df = df[df["Cluster"] == i]

            st.subheader(f"Cluster {i} - {CLUSTER_LABEL[i]}")
            st.write(f"Jumlah Wilayah: {cluster_df['kabupaten_kota'].nunique()}")
            st.write(f"PDRB Rata-rata: Rp {cluster_df['PDRB'].mean():,.0f}")
            st.write(
                f"Penduduk Miskin Rata-rata: "
                f"{cluster_df['jumlah_penduduk_miskin'].mean():.1f} ribu"
            )

            with st.expander("Lihat Wilayah"):
                for w in cluster_df["kabupaten_kota"].unique():
                    st.write(f"• {w}")

            st.divider()

    # ========== DATABASE ==========
    elif menu == "Dataset":
        st.title("📋 Dataset")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            "hasil_clustering.csv",
            "text/csv"
        )


# ==================== RUN ====================
if __name__ == "__main__":
    main()
