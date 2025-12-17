import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Clustering Kemiskinan Jawa Barat",
    layout="wide"
)

# ====================================
# LOAD MODEL + DATA
# ====================================
@st.cache_resource
def load_model():
    return joblib.load("kmeans.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_kemiskinan.csv")
    return df

model_bundle = load_model()
df = load_data()

kmeans = model_bundle["model"]
scaler = model_bundle["scaler"]
features = model_bundle["features"]

# ====================================
# PREPROCESS
# ====================================
X = df[features]
X_scaled = scaler.transform(X)
df["Cluster"] = kmeans.predict(X_scaled)

cluster_data = {
    "skor_silhouette": float(model_bundle["silhouette"]),
    "skor_davies_bouldin": float(model_bundle["davies_bouldin"])
}

# ====================================
# SIDEBAR
# ====================================
st.sidebar.title("Menu Analisis")
menu = st.sidebar.radio(
    "Pilih Analisis:",
    ["Dashboard", "EDA & Visualisasi", "Hasil Clustering", "Dataset", "Insight & Rekomendasi"]
)

# ====================================
# DASHBOARD
# ====================================
if menu == "Dashboard":
    st.title("Dashboard Clustering Kemiskinan Jawa Barat")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Silhouette Score",
            f"{cluster_data['skor_silhouette']:.3f}"
        )

    with col2:
        st.metric(
            "Davies-Bouldin Index",
            f"{cluster_data['skor_davies_bouldin']:.3f}"
        )

# ====================================
# EDA
# ====================================
elif menu == "EDA & Visualisasi":
    st.title("EDA & Visualisasi")

    fitur = st.selectbox("Pilih Fitur", features)

    fig, ax = plt.subplots()
    df[fitur].hist(ax=ax)
    st.pyplot(fig)

# ====================================
# HASIL CLUSTERING
# ====================================
elif menu == "Hasil Clustering":
    st.title("Hasil Clustering")

    st.dataframe(df)

    fig, ax = plt.subplots()
    df["Cluster"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# ====================================
# DATASET
# ====================================
elif menu == "Dataset":
    st.title("Dataset")
    st.dataframe(df)

# ====================================
# INSIGHT
# ====================================
elif menu == "Insight & Rekomendasi":
    st.title("Insight & Rekomendasi")

    st.markdown("""
    - Cluster dengan jumlah penduduk miskin dan pengangguran tinggi perlu prioritas bantuan.
    - Wilayah dengan PDRB tinggi namun kemiskinan tinggi menandakan ketimpangan ekonomi.
    - Hasil clustering dapat menjadi dasar kebijakan berbasis data.
    """)

