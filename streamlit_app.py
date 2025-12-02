import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

st.title("Clustering Daerah Rawan Kemiskinan")
st.write("Aplikasi ini mengelompokkan daerah menjadi kategori *Rawan Kemiskinan* dan *Aman* berdasarkan model K-Means.")

# ------------------------------------------------------------
# 1. Load model + scaler dari kmeans.pkl
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
    file = "datasetkemiskinan.csv"
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

    # Ambil hanya kolom numerik
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if numeric_df.shape[1] < 2:
        st.error("Dataset membutuhkan minimal 2 kolom numerik.")
        st.stop()

    # ------------------------------------------------------------
    # FIX paling penting:
    # Samakan kolom dataset dengan kolom saat scaler training
    # ------------------------------------------------------------
    try:
        numeric_df = numeric_df[scaler.feature_names_in_]
    except Exception as e:
        st.error(
            "Kolom dataset tidak cocok dengan model! "
            "Dataset harus memiliki kolom numerik berikut (urutan wajib sama): "
            + ", ".join(scaler.feature_names_in_)
        )
        st.stop()

    # ------------------------------------------------------------
    # 5. Scaling data
    # ------------------------------------------------------------
    scaled = scaler.transform(numeric_df)

    # ------------------------------------------------------------
    # 6. Prediksi cluster
    # ------------------------------------------------------------
    df["cluster"] = model.predict(scaled)

    # Hitung rata-rata untuk menentukan cluster risiko tertinggi
    cluster_means = df.groupby("cluster")[numeric_df.columns].mean()
    risk_scores = cluster_means.mean(axis=1)
    highest_risk_cluster = risk_scores.idxmax()

    # Tambahkan kategori
    df["kategori"] = df["cluster"].apply(
        lambda c: "Rawan Kemiskinan" if c == highest_risk_cluster else "Aman"
    )

    # ------------------------------------------------------------
    # 7. Tampilkan Hasil
    # ------------------------------------------------------------
    st.subheader("Kategori Daerah")
    st.write(df)

    # Daerah Rawan
    st.subheader("Daerah Rawan Kemiskinan (🟥)")
    st.write(df[df["kategori"] == "Rawan Kemiskinan"])

    # Daerah Aman
    st.subheader("Daerah Aman (🟩)")
    st.write(df[df["kategori"] == "Aman"])

    # ------------------------------------------------------------
    # 8. Visualisasi Cluster
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

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Rawan Kemiskinan',
                   markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Aman',
                   markerfacecolor='green', markersize=10)
    ]
    ax.legend(handles=handles)

    st.pyplot(fig)

    # ------------------------------------------------------------
    # 9. Tombol Download
    # ------------------------------------------------------------
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Hasil (CSV)",
        data=csv_out,
        file_name="hasil_cluster_kemiskinan.csv",
        mime="text/csv"
    )

