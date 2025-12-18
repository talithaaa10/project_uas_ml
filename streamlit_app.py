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
    return bundle["model"], bundle["scaler"], bundle.get("kmeans_model", None)

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
    df["cluster_kmeans"] = model.predict(X_scaled)
    
    return df

def build_cluster_data(df):
    """Membangun data cluster dengan kategori"""
    cluster_data = {
        "skor_silhouette": 0.0,
        "skor_davies_bouldin": 0.0,
        "cluster": {}
    }

    # Tentukan kategori berdasarkan karakteristik cluster
    kategori_cluster = {
        0: "Wilayah Ekonomi Tinggi",
        1: "Wilayah Ekonomi Menengah", 
        2: "Wilayah Perlu Perhatian"
    }

    for i in sorted(df["cluster_kmeans"].unique()):
        cdf = df[df["cluster_kmeans"] == i]

        cluster_data["cluster"][str(i)] = {
            "kategori": kategori_cluster[i],
            "jumlah": len(cdf),
            "pdrb_rata": cdf["PDRB"].mean(),
            "miskin_rata": cdf["jumlah_penduduk_miskin"].mean(),
            "garis_kemiskinan_rata": cdf["garis_kemiskinan"].mean(),
            "pengangguran_rata": cdf["jumlah_pengangguran"].mean(),
            "contoh_wilayah": cdf["kabupaten_kota"].unique()[:5].tolist(),
            "pdrb_max": cdf["PDRB"].max(),
            "pdrb_min": cdf["PDRB"].min()
        }

    return cluster_data

def create_line_plot(df_raw):
    """Create line plot for yearly trends"""
    annual_stats = df_raw.groupby('Tahun').agg({
        'jumlah_penduduk_miskin': 'sum',
        'garis_kemiskinan': 'mean'
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(annual_stats['Tahun'], annual_stats['jumlah_penduduk_miskin'], 
             marker='o', color='red', linewidth=2.5)
    ax1.set_title('Tren Total Penduduk Miskin (2017-2019)')
    ax1.set_ylabel('Jumlah Penduduk Miskin (Jiwa)')
    ax1.set_xlabel('Tahun')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(annual_stats['Tahun'], annual_stats['garis_kemiskinan'], 
             marker='o', color='blue', linewidth=2.5)
    ax2.set_title('Tren Rata-rata Garis Kemiskinan (2017-2019)')
    ax2.set_ylabel('Garis Kemiskinan (Rupiah)')
    ax2.set_xlabel('Tahun')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_box_plots(df_raw):
    """Create box plots for distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.boxplot(data=df_raw, x='Tahun', y='jumlah_penduduk_miskin', ax=ax1)
    ax1.set_title('Distribusi Penduduk Miskin per Tahun')
    ax1.set_ylabel('Jumlah Penduduk Miskin (Jiwa)')
    
    sns.boxplot(data=df_raw, x='Tahun', y='jumlah_pengangguran', ax=ax2)
    ax2.set_title('Distribusi Pengangguran per Tahun')
    ax2.set_ylabel('Jumlah Pengangguran (Jiwa)')
    
    plt.tight_layout()
    return fig

def create_scatter_plot(df_raw):
    """Create scatter plot PDRB vs Kemiskinan"""
    df_2019 = df_raw[df_raw['Tahun'] == 2019].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_2019['PDRB'], df_2019['jumlah_penduduk_miskin'],
                        s=df_2019['jumlah_warga_jabar']/10,
                        alpha=0.6, cmap='viridis')
    
    ax.set_xlabel('PDRB (Ekonomi Makro)')
    ax.set_ylabel('Jumlah Penduduk Miskin')
    ax.set_title('Peta Sebaran: Ekonomi vs Kemiskinan (2019)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_correlation_matrix(df_raw):
    """Create correlation matrix heatmap"""
    df_2019 = df_raw[df_raw['Tahun'] == 2019].copy()
    kolom_korelasi = ['jumlah_warga_jabar', 'jumlah_penduduk_miskin', 
                      'garis_kemiskinan', 'jumlah_pengangguran', 'PDRB']
    
    corr_matrix = df_2019[kolom_korelasi].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Matriks Korelasi Variabel (2019)')
    
    plt.tight_layout()
    return fig

def create_cluster_comparison(cluster_data):
    """Create bar chart comparing clusters"""
    clusters = ['Cluster 0', 'Cluster 1', 'Cluster 2']
    pdrb_values = [cluster_data['cluster'][str(i)]['pdrb_rata'] for i in range(3)]
    miskin_values = [cluster_data['cluster'][str(i)]['miskin_rata'] for i in range(3)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    bars1 = ax1.bar(clusters, pdrb_values, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax1.set_ylabel('PDRB Rata-rata (Rp)')
    ax1.set_title('Perbandingan PDRB per Cluster')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'Rp {height:,.0f}', ha='center', va='bottom', fontsize=9)
    
    bars2 = ax2.bar(clusters, miskin_values, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax2.set_ylabel('Penduduk Miskin Rata-rata (ribu)')
    ax2.set_title('Perbandingan Penduduk Miskin per Cluster')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_kmeans_scatter_plot(df, cluster_data):
    """Create scatter plot for KMeans clustering results"""
    # Warna untuk tiap cluster
    cluster_colors = {
        0: '#2E86AB',  # Biru untuk cluster 0
        1: '#A23B72',  # Ungu untuk cluster 1
        2: '#F18F01'   # Oranye untuk cluster 2
    }
    
    # Normalisasi sementara untuk visualisasi
    df['PDRB_norm'] = (df['PDRB'] - df['PDRB'].min()) / (df['PDRB'].max() - df['PDRB'].min()) * 100
    df['miskin_norm'] = (df['jumlah_penduduk_miskin'] - df['jumlah_penduduk_miskin'].min()) / (df['jumlah_penduduk_miskin'].max() - df['jumlah_penduduk_miskin'].min()) * 100
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot data points per cluster
    for cluster_num in df['cluster_kmeans'].unique():
        cluster_df = df[df['cluster_kmeans'] == cluster_num]
        color = cluster_colors[cluster_num]
        
        ax.scatter(cluster_df['PDRB_norm'], cluster_df['miskin_norm'],
                  c=color, s=100, alpha=0.7,
                  label=f'Cluster {cluster_num}: {cluster_data["cluster"][str(cluster_num)]["kategori"]}',
                  edgecolors='white', linewidth=0.5)
        
        # Plot centroid
        centroid_x = cluster_df['PDRB_norm'].mean()
        centroid_y = cluster_df['miskin_norm'].mean()
        
        ax.scatter(centroid_x, centroid_y,
                  c=color, s=300, marker='X',
                  edgecolors='black', linewidth=2,
                  label=f'Centroid Cluster {cluster_num}')
    
    # Annotate some sample points
    for idx, row in df.sample(min(10, len(df))).iterrows():
        ax.annotate(row['kabupaten_kota'][:10],  # Ambil 10 karakter pertama
                   (row['PDRB_norm'], row['miskin_norm']),
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel('PDRB (Normalized)')
    ax.set_ylabel('Penduduk Miskin (Normalized)')
    ax.set_title('Visualisasi Hasil Clustering K-Means', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Tambah informasi cluster
    info_text = "Informasi Cluster:\n"
    for i in range(3):
        info_text += f"\nCluster {i}: {cluster_data['cluster'][str(i)]['kategori']}"
        info_text += f" ({cluster_data['cluster'][str(i)]['jumlah']} wilayah)"
    
    # Tambah text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.05, 0.5, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=props)
    
    plt.tight_layout()
    
    # Hapus kolom normalisasi sementara
    df.drop(['PDRB_norm', 'miskin_norm'], axis=1, inplace=True)
    
    return fig

def create_feature_scatter_plot(df, feature_x, feature_y, cluster_data):
    """Create scatter plot based on specific features"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Warna untuk tiap cluster
    cluster_colors = {
        0: '#2E86AB',
        1: '#A23B72',
        2: '#F18F01'
    }
    
    # Plot per cluster
    for cluster_num in df['cluster_kmeans'].unique():
        cluster_df = df[df['cluster_kmeans'] == cluster_num]
        color = cluster_colors[cluster_num]
        
        scatter = ax.scatter(cluster_df[feature_x], cluster_df[feature_y],
                           c=color, s=150, alpha=0.7,
                           label=f'Cluster {cluster_num}',
                           edgecolors='white', linewidth=1)
    
    ax.set_xlabel(feature_x.replace('_', ' ').title())
    ax.set_ylabel(feature_y.replace('_', ' ').title())
    ax.set_title(f'Clustering berdasarkan {feature_x} vs {feature_y}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Cluster')
    
    # Tambah informasi rata-rata per cluster
    info_text = "Rata-rata per Cluster:\n"
    for i in range(3):
        c = cluster_data['cluster'][str(i)]
        info_text += f"\nCluster {i}:\n"
        info_text += f"  {feature_x}: {df[df['cluster_kmeans']==i][feature_x].mean():.2f}\n"
        info_text += f"  {feature_y}: {df[df['cluster_kmeans']==i][feature_y].mean():.2f}"
    
    plt.figtext(0.02, 0.02, info_text, fontsize=9, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    return fig

# ==================== MAIN APP ====================
def main():
    # Load data and apply clustering
    df, df_raw = load_data()
    model, scaler, kmeans_model = load_model()
    df = apply_clustering(df, model, scaler)
    
    cluster_data = build_cluster_data(df)
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.title("📊 Menu Analisis")
        
        menu = st.radio(
            "Pilih Analisis:",
            ["🏠 Dashboard", "📈 EDA & Visualisasi", "🎯 Hasil Clustering", "📋 Dataset"]
        )
        
        st.divider()
        st.caption("Proyek Clustering Kemiskinan")
        st.caption("Jawa Barat 2017-2019")
    
    # ===== DASHBOARD =====
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
        
        # Cluster Overview dengan kategori
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
                st.markdown(f"**Kategori:** {c['kategori']}")
                
                st.metric("Jumlah Wilayah", c['jumlah'])
                st.write(f"*PDRB Rata:* Rp {c['pdrb_rata']:,.0f}")
                st.write(f"*Penduduk Miskin:* {c['miskin_rata']:.1f} ribu")
                
                with st.expander("Lihat contoh wilayah"):
                    for wilayah in c['contoh_wilayah']:
                        st.write(f"• {wilayah}")
    
    # ===== EDA & VISUALISASI =====
    elif menu == "📈 EDA & Visualisasi":
        st.title("📈 EXPLORATORY DATA ANALYSIS")
        st.markdown("*Analisis Data dan Visualisasi Indikator Kemiskinan*")
        
        tab1, tab2 = st.tabs(["📈 Trend", "🎯 Scatter K-Means"])
        
        with tab1:
            st.subheader("Analisis Trend Tahun 2017-2019")
            st.pyplot(create_line_plot(df_raw))
            st.caption("Gambar 1: Trend penduduk miskin dan garis kemiskinan")
            
        with tab2:
            st.subheader("🎯 Visualisasi Hasil Clustering K-Means")
            
            # Pilih jenis visualisasi
            viz_type = st.radio(
                "Pilih Jenis Visualisasi:",
                ["Scatter Plot PDRB vs Kemiskinan"]
            )
            
            if viz_type == "Scatter Plot PDRB vs Kemiskinan":
                fig = create_kmeans_scatter_plot(df.copy(), cluster_data)
                st.pyplot(fig)
                st.caption("Visualisasi hasil clustering K-Means: PDRB vs Penduduk Miskin")
                
            
    # ===== HASIL CLUSTERING =====
    elif menu == "🎯 Hasil Clustering":
        st.title("🎯 ANALISIS HASIL CLUSTERING")
        st.markdown("*Analisis Mendalam Tiap Cluster*")
        
        # Tab untuk berbagai visualisasi
        tab1, tab2 = st.tabs(["📊 Perbandingan", "🔍 Detail Cluster"])
        
        with tab1:
            st.subheader("📊 Perbandingan Antar Cluster")
            fig = create_kmeans_scatter_plot(df.copy(), cluster_data)
            st.pyplot(fig)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Wilayah", len(df))
            with col2:
                st.metric("Jumlah Cluster", "3")
            with col3:
                # Hitung silhouette score sederhana (jika tersedia)
                if cluster_data.get('skor_silhouette'):
                    st.metric("Silhouette Score", f"{cluster_data['skor_silhouette']:.3f}")
                else:
                    st.metric("Cluster Terbesar", 
                             f"Cluster {max(cluster_data['cluster'].items(), key=lambda x: x[1]['jumlah'])[0]}")

            st.subheader("📊 Perbandingan Numerik Cluster")
            st.pyplot(create_cluster_comparison(cluster_data))

        
        with tab2:
            st.subheader("🔍 Analisis Detail Tiap Cluster")
            
            for i in range(3):
                c = cluster_data['cluster'][str(i)]
                
                # Header dengan kategori
                if i == 0:
                    st.markdown(f"<h3 style='color:#2E86AB;'>🔵 CLUSTER {i}: {c['kategori']}</h3>", 
                               unsafe_allow_html=True)
                elif i == 1:
                    st.markdown(f"<h3 style='color:#A23B72;'>🟣 CLUSTER {i}: {c['kategori']}</h3>", 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color:#F18F01;'>🟠 CLUSTER {i}: {c['kategori']}</h3>", 
                               unsafe_allow_html=True)
                
                # Metrics
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
                st.write("📍 **Daftar Wilayah:**")
                cluster_regions = df[df['cluster_kmeans'] == i]['kabupaten_kota'].unique()
                
                cols_regions = st.columns(3)
                regions_per_col = len(cluster_regions) // 3 + 1
                
                for col_idx, col in enumerate(cols_regions):
                    with col:
                        start_idx = col_idx * regions_per_col
                        end_idx = min((col_idx + 1) * regions_per_col, len(cluster_regions))
                        
                        for region in cluster_regions[start_idx:end_idx]:
                            st.write(f"• {region}")
                
                st.divider()
    
    # ===== DATASET =====
    else:
        st.title("📋 DATASET LENGKAP")
        st.markdown("*Tabel Data dengan Filter dan Download*")
        
        # Pilih kolom yang akan ditampilkan
        selected_columns = [
            'Tahun',
            'kabupaten_kota',
            'jumlah_warga_jabar',
            'jumlah_penduduk_miskin',
            'garis_kemiskinan',
            'jumlah_pengangguran',
            'PDRB',
            'cluster_kmeans'
        ]
        
        # Filters
        st.subheader("🔍 Filter Data")
        
        col1, col2 = st.columns(2)
        with col1:
            tahun_filter = st.multiselect("Pilih Tahun", 
                                         options=sorted(df['Tahun'].unique()), 
                                         default=sorted(df['Tahun'].unique()))
        with col2:
            cluster_filter = st.multiselect("Pilih Cluster", 
                                           options=sorted(df['cluster_kmeans'].unique()), 
                                           default=sorted(df['cluster_kmeans'].unique()))
        
        # Apply filters
        filtered_df = df.copy()
        if tahun_filter:
            filtered_df = filtered_df[filtered_df['Tahun'].isin(tahun_filter)]
        if cluster_filter:
            filtered_df = filtered_df[filtered_df['cluster_kmeans'].isin(cluster_filter)]
        
        # Pilih hanya kolom yang diinginkan
        filtered_df = filtered_df[selected_columns]
        
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
        
        # Format nama kolom untuk tampilan yang lebih baik
        display_columns = {
            'Tahun': 'Tahun',
            'kabupaten_kota': 'Kabupaten/Kota',
            'jumlah_warga_jabar': 'Jumlah Warga',
            'jumlah_penduduk_miskin': 'Penduduk Miskin',
            'garis_kemiskinan': 'Garis Kemiskinan',
            'jumlah_pengangguran': 'Pengangguran',
            'PDRB': 'PDRB',
            'cluster_kmeans': 'Cluster K-Means'
        }
        
        st.dataframe(
            filtered_df.rename(columns=display_columns),
            use_container_width=True,
            column_config={
                "Tahun": st.column_config.NumberColumn(format="%d"),
                "PDRB": st.column_config.NumberColumn(format="Rp %,.0f"),
                "Garis Kemiskinan": st.column_config.NumberColumn(format="Rp %,.0f"),
                "Cluster K-Means": st.column_config.NumberColumn(format="%d")
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








