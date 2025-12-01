import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

st.title("Clustering Daerah Rawan Kemiskinan")
st.write("Aplikasi ini mengelompokkan daerah menjadi kategori *Rawan Kemiskinan* dan *Aman* berdasarkan model K-Means.")

# ------------------------------------------------------------
# 1. Load model + scaler dari model.pkl
# ------------------------------------------------------------
@st.cache_resource
def load_bundle():
    bundle = joblib.load("kmeans.pkl")
    model = bundle["model"]      # KMeans
    scaler = bundle["scaler"]    # StandardScaler
    return model, scaler

model, scaler = load_bundle()

# ------------------------------------------------------------
# 2. Auto-load CSV default
# ------------------------------------------------------------
def load_default_csv():
    file = "dataset_kemiskinan.csv"
    if os.path.exists(file):
        st.success(f"File default ditemukan dan dimuat otomatis: {file}")
        return pd.read_csv(file)
    st.warning("File default tidak ditemukan.")
    return None

df = load_default_csv()

# ------------------------------------------------------------
# 3. Upload CSV Opsional
# ------------------------------------------------------------
uploaded = st.file_uploader("Upload file CSV (opsional):", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.info("File upload digunakan menggantikan file default.")

# ------------------------------------------------------------
# 4. Lanjut jika ada dataset
# ------------------------------------------------------------
if df is not None:
    st.subheader("Preview Data")
    st.write(df.head())

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if numeric_df.shape[1] < 2:
        st.error("Dataset membutuhkan minimal 2 kolom numerik.")
    else:
        # Scaling
        scaled = scaler.transform(numeric_df)

        # Prediksi cluster
        df["cluster"] = model.predict(scaled)

        # ------------------------------------------------------------
        # 5. Menentukan cluster dengan risiko tertinggi
        # ------------------------------------------------------------
        cluster_means = df.groupby("cluster")[numeric_df.columns].mean()

        # Asumsi: cluster dengan rata-rata lebih tinggi dianggap lebih rawan kemiskinan
        risk_scores = cluster_means.mean(axis=1)
        highest_risk_cluster = risk_scores.idxmax()

        # Tambahkan kolom kategori
        df["kategori"] = df["cluster"].apply(
            lambda c: "Rawan Kemiskinan" if c == highest_risk_cluster else "Aman"
        )

        # ------------------------------------------------------------
        # 6. Tampilkan hasil kategori
        # ------------------------------------------------------------
        st.subheader("Kategori Daerah")
        st.write(df[["cluster", "kategori"] + list(df.columns[:-2])])

        # ------------------------------------------------------------
        # 7. Tampilkan daftar daerah rawan dan aman
        # ------------------------------------------------------------
        st.subheader("Daerah Rawan Kemiskinan (🟥)")
        st.write(df[df["kategori"] == "Rawan Kemiskinan"])

        st.subheader("Daerah Aman (🟩)")
        st.write(df[df["kategori"] == "Aman"])

        # ------------------------------------------------------------
        # 8. Visualisasi Cluster dengan Kategori
        # ------------------------------------------------------------
        st.subheader("Visualisasi Clustering")

        f1 = numeric_df.columns[0]
        f2 = numeric_df.columns[1]

        fig, ax = plt.subplots()

        colors = df["kategori"].map({
            "Rawan Kemiskinan": "red",
            "Aman": "green"
        })

        ax.scatter(
            numeric_df[f1],
            numeric_df[f2],
            c=colors,
            s=60
        )

        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title("Visualisasi Daerah Rawan Kemiskinan vs Aman")

        # Legend manual
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label='Rawan Kemiskinan', markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Aman', markerfacecolor='green', markersize=10)
        ]
        ax.legend(handles=handles)

        st.pyplot(fig)

        # ------------------------------------------------------------
        # 9. Download hasil
        # ------------------------------------------------------------
        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Hasil (CSV)",
            data=csv_out,
            file_name="hasil_cluster_kemiskinan.csv",
            mime="text/csv"

        )

