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
            ["🏠 Dashboard", "📈 EDA & Visualisasi", "🎯 Hasil Clustering", "📋 Dataset"]
        )

   if menu == "🏠 Dashboard":
        st.title("📊 DASHBOARD ANALISIS CLUSTERING")
        st.markdown("*Segmentasi Wilayah Jawa Barat Berdasarkan Indikator Kemiskinan*")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Data", len(df))
        with col2:
            st.metric("Jumlah Wilayah", df['kabupaten_kota'].nunique())
        with col3:
            st.metric("Periode Data", f"{df['Tahun'].min()}-{df['Tahun'].max()}")
        with col4:
            st.metric("Jumlah Cluster", "3")
        
        st.divider()
        
        # Cluster Overview
        st.subheader("🎯 HASIL CLUSTERING")
        
        cols = st.columns(3)
        for i, col in enumerate(cols):
            with col:
                c = cluster_data['cluster'][str(i)]
                if i == 0:
                    color = "#2E86AB"
                elif i == 1:
                    color = "#A23B72"
                else:
                    color = "#F18F01"
                
                st.markdown(f"<h3 style='color:{color};'>CLUSTER {i}</h3>", unsafe_allow_html=True)
                st.markdown(f"{c['kategori']}")
                
                st.metric("Jumlah Wilayah", c['jumlah'])
                st.write(f"*PDRB Rata:* Rp {c['pdrb_rata']:,.0f}")
                st.write(f"*Penduduk Miskin:* {c['miskin_rata']:.1f} ribu")
                
                with st.expander("Lihat contoh wilayah"):
                    for wilayah in c['contoh_wilayah']:
                        st.write(f"• {wilayah}")
        
        # Quick Visual
        st.divider()
        st.subheader("📊 Visualisasi Cepat")
        
        fig = create_cluster_comparison(cluster_data)
        st.pyplot(fig)
        st.caption("Perbandingan PDRB dan Penduduk Miskin antar Cluster")


    # ===== EDA & VISUALISASI =====
    elif menu == "📈 EDA & Visualisasi":
        st.title("📈 EXPLORATORY DATA ANALYSIS")
        st.markdown("*Analisis Data dan Visualisasi Indikator Kemiskinan*")
        
        tab1, tab2, tab3 = st.tabs(["📈 Trend", "📊 Distribusi", "🔗 Korelasi"])
        
        with tab1:
            st.subheader("Analisis Trend Tahun 2017-2019")
            st.pyplot(create_line_plot(df_raw))
            st.caption("Gambar 1: Trend penduduk miskin dan garis kemiskinan")
        
        with tab2:
            st.subheader("Analisis Distribusi Data")
            col1, col2 = st.columns(2)
            
            with col1:
                st.pyplot(create_box_plots(df_raw))
                st.caption("Gambar 2: Distribusi penduduk miskin dan pengangguran")
            
            with col2:
                st.pyplot(create_scatter_plot(df_raw))
                st.caption("Gambar 3: Hubungan PDRB vs Penduduk Miskin (2019)")
        
        with tab3:
            st.subheader("Analisis Korelasi Antar Variabel")
            st.pyplot(create_correlation_matrix(df_raw))
            st.caption("Gambar 4: Matriks korelasi indikator kemiskinan (2019)")
            
            # Correlation insights
            st.subheader("💡 Insight Korelasi")
            st.info("""
            *Interpretasi Korelasi:*
            - *PDRB vs Penduduk Miskin*: Korelasi negatif (-0.xx)
              → Semakin tinggi PDRB, cenderung semakin rendah penduduk miskin
            - *Pengangguran vs Penduduk Miskin*: Korelasi positif (+0.xx)
              → Semakin tinggi pengangguran, semakin tinggi penduduk miskin
            - *Garis Kemiskinan vs PDRB*: Korelasi positif (+0.xx)
              → Wilayah dengan PDRB tinggi cenderung memiliki garis kemiskinan tinggi
            """)


    elif menu == "🎯 Hasil Clustering":
        st.title("🎯 ANALISIS HASIL CLUSTERING")
        st.markdown("*Analisis Mendalam Tiap Cluster*")
        
        # Cluster Comparison
        st.subheader("📊 Perbandingan Antar Cluster")
        st.pyplot(create_cluster_comparison(cluster_data))
        
        # Detailed Analysis per Cluster
        st.subheader("🔍 Analisis Detail Tiap Cluster")
        
        for i in range(3):
            c = cluster_data['cluster'][str(i)]
            
            if i == 0:
                st.markdown(f"<h3 style='color:#2E86AB;'>🔵 CLUSTER {i}: {c['kategori']}</h3>", 
                           unsafe_allow_html=True)
            elif i == 1:
                st.markdown(f"<h3 style='color:#A23B72;'>🟣 CLUSTER {i}: {c['kategori']}</h3>", 
                           unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:#F18F01;'>🟠 CLUSTER {i}: {c['kategori']}</h3>", 
                           unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Jumlah Wilayah", c['jumlah'])
                st.write(f"*PDRB Rata-rata:* Rp {c['pdrb_rata']:,.0f}")
                st.write(f"*Garis Kemiskinan:* Rp {c['garis_kemiskinan_rata']:,.0f}")
            
            with col2:
                st.metric("Penduduk Miskin", f"{c['miskin_rata']:.1f} ribu")
                st.write(f"*Pengangguran:* {c['pengangguran_rata']:,.0f} jiwa")
                st.write(f"*Persentase Data:* {(c['jumlah']/len(df)*100):.1f}%")
            
            # Wilayah dalam cluster
            st.write("📍 Daftar Wilayah:")
            cluster_regions = df[df['Cluster'] == i]['kabupaten_kota'].unique()
            
            cols_regions = st.columns(3)
            regions_per_col = len(cluster_regions) // 3 + 1
            
            for col_idx, col in enumerate(cols_regions):
                with col:
                    start_idx = col_idx * regions_per_col
                    end_idx = min((col_idx + 1) * regions_per_col, len(cluster_regions))
                    
                    for region in cluster_regions[start_idx:end_idx]:
                        st.write(f"• {region}")
            
            st.divider()

  # ===== DATABASE =====
    else:
        st.title("📋 DATASET LENGKAP")
        st.markdown("*Tabel Data dengan Filter dan Download*")
        
        # Filters
        st.subheader("🔍 Filter Data")
        
        col1, col2 = st.columns(2)
        with col1:
            tahun_filter = st.multiselect("Pilih Tahun", 
                                         options=sorted(df['Tahun'].unique()), 
                                         default=sorted(df['Tahun'].unique()))
        with col2:
            cluster_filter = st.multiselect("Pilih Cluster", 
                                           options=sorted(df['Cluster'].unique()), 
                                           default=sorted(df['Cluster'].unique()))
        
        # Apply filters
        filtered_df = df.copy()
        if tahun_filter:
            filtered_df = filtered_df[filtered_df['Tahun'].isin(tahun_filter)]
        if cluster_filter:
            filtered_df = filtered_df[filtered_df['Cluster'].isin(cluster_filter)]
        
        # Statistics
        st.subheader("📊 Statistik Data")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records", len(filtered_df))
        with col2:
            avg_pdrb = filtered_df['PDRB'].mean()
            st.metric("PDRB Rata", f"Rp {avg_pdrb:,.0f}")
        with col3:
            avg_miskin = filtered_df['jumlah_penduduk_miskin'].mean()
            st.metric("Penduduk Miskin Rata", f"{avg_miskin:.1f} ribu")
        
        # Data Table
        st.subheader("📋 Tabel Data")
        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config={
                "Tahun": st.column_config.NumberColumn(format="%d"),
                "PDRB": st.column_config.NumberColumn(format="Rp %,.0f"),
                "garis_kemiskinan": st.column_config.NumberColumn(format="Rp %,.0f"),
                "Cluster": st.column_config.NumberColumn(format="%d")
            },
            hide_index=True
        )
        
        # Download
        st.subheader("💾 Download Data")
        
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data Terfilter (CSV)",
            data=csv,
            file_name="data_clustering_kemiskinan.csv",
            mime="text/csv"
        )
    

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()






