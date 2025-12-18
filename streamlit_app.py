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
    df["Cluster"] = model.predict(X_scaled)
    
    # Simpan koordinat untuk visualisasi
    df["PCA1"], df["PCA2"] = get_pca_coordinates(X_scaled)
    
    return df

def get_pca_coordinates(X_scaled):
    """Reduksi dimensi sederhana untuk visualisasi 2D"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X_scaled)
    return coordinates[:, 0], coordinates[:, 1]

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

    for i in sorted(df["Cluster"].unique()):
        cdf = df[df["Cluster"] == i]

        cluster_data["cluster"][str(i)] = {
            "kategori": kategori_cluster[i],
            "jumlah": len(cdf),
            "pdrb_rata": cdf["PDRB"].mean(),
            "miskin_rata": cdf["jumlah_penduduk_miskin"].mean(),
            "garis_kemiskinan_rata": cdf["garis_kemiskinan"].mean(),
            "pengangguran_rata": cdf["jumlah_pengangguran"].mean(),
            "contoh_wilayah": cdf["kabupaten_kota"].unique()[:5].tolist(),
            "centroid_pca1": cdf["PCA1"].mean(),
            "centroid_pca2": cdf["PCA2"].mean()
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
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot data points per cluster
    for cluster_num in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster_num]
        color = cluster_colors[cluster_num]
        
        ax.scatter(cluster_df['PCA1'], cluster_df['PCA2'],
                  c=color, s=100, alpha=0.7,
                  label=f'Cluster {cluster_num}: {cluster_data["cluster"][str(cluster_num)]["kategori"]}',
                  edgecolors='white', linewidth=0.5)
        
        # Plot centroid
        centroid_x = cluster_data['cluster'][str(cluster_num)]['centroid_pca1']
        centroid_y = cluster_data['cluster'][str(cluster_num)]['centroid_pca2']
        
        ax.scatter(centroid_x, centroid_y,
                  c=color, s=300, marker='X',
                  edgecolors='black', linewidth=2,
                  label=f'Centroid Cluster {cluster_num}')
    
    # Annotate some sample points
    for idx, row in df.sample(min(10, len(df))).iterrows():
        ax.annotate(row['kabupaten_kota'][:10],  # Ambil 10 karakter pertama
                   (row['PCA1'], row['PCA2']),
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Komponen PCA 1')
    ax.set_ylabel('Komponen PCA 2')
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
    return fig

def create_3d_kmeans_plot(df, cluster_data):
    """Create 3D scatter plot for KMeans clustering"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Warna untuk tiap cluster
    cluster_colors = {
        0: '#2E86AB',  # Biru
        1: '#A23B72',  # Ungu
        2: '#F18F01'   # Oranye
    }
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot data points per cluster
    for cluster_num in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster_num]
        color = cluster_colors[cluster_num]
        
        ax.scatter(cluster_df['PCA1'], cluster_df['PCA2'], cluster_df['PDRB_normalized'],
                  c=color, s=80, alpha=0.7,
                  label=f'Cluster {cluster_num}: {cluster_data["cluster"][str(cluster_num)]["kategori"]}',
                  edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PDRB (Normalized)')
    ax.set_title('Visualisasi 3D Hasil Clustering K-Means', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Rotate untuk view yang lebih baik
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
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
    for cluster_num in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster_num]
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
        info_text += f"  {feature_x}: {df[df['Cluster']==i][feature_x].mean():.2f}\n"
        info_text += f"  {feature_y}: {df[df['Cluster']==i][feature_y].mean():.2f}"
    
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
    
    # Normalize PDRB for 3D plot
    df['PDRB_normalized'] = (df['PDRB'] - df['PDRB'].min()) / (df['PDRB'].max() - df['PDRB'].min())
    
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
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend", "📊 Distribusi", "🔗 Korelasi", "🎯 Scatter K-Means"])
        
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
        
        with tab4:
            st.subheader("🎯 Visualisasi Hasil Clustering K-Means")
            
            st.info("""
            **Penjelasan Visualisasi:**
            - Setiap titik mewakili satu wilayah kabupaten/kota
            - Warna menunjukkan cluster hasil K-Means
            - Tanda 'X' menunjukkan centroid (pusat) tiap cluster
            - PCA digunakan untuk mereduksi data multi-dimensi menjadi 2D untuk visualisasi
            """)
            
            # Pilih jenis visualisasi
            viz_type = st.radio(
                "Pilih Jenis Visualisasi:",
                ["Scatter Plot 2D", "Scatter Plot 3D", "Plot berdasarkan Fitur Tertentu"]
            )
            
            if viz_type == "Scatter Plot 2D":
                fig = create_kmeans_scatter_plot(df, cluster_data)
                st.pyplot(fig)
                st.caption("Visualisasi 2D hasil clustering K-Means dengan PCA")
                
                # Insight dari visualisasi
                with st.expander("📊 Insight dari Visualisasi"):
                    st.write("""
                    **Interpretasi Visual:**
                    1. **Cluster Separation**: Jarak antar centroid menunjukkan seberapa berbeda karakteristik tiap cluster
                    2. **Cluster Density**: Kepadatan titik dalam cluster menunjukkan homogenitas wilayah
                    3. **Outliers**: Titik yang jauh dari centroid mungkin memiliki karakteristik unik
                    4. **Cluster Shape**: Pola sebaran titik menunjukkan karakteristik distribusi
                    """)
            
            elif viz_type == "Scatter Plot 3D":
                fig = create_3d_kmeans_plot(df, cluster_data)
                st.pyplot(fig)
                st.caption("Visualisasi 3D hasil clustering K-Means")
                
                st.info("""
                **Tips Interaksi 3D:**
                - Klik dan drag untuk memutar plot 3D
                - Zoom dengan scroll mouse
                - Pandangan 3D membantu memahami struktur data yang kompleks
                """)
            
            else:  # Plot berdasarkan Fitur Tertentu
                col1, col2 = st.columns(2)
                with col1:
                    feature_x = st.selectbox(
                        "Pilih fitur untuk sumbu X:",
                        ['jumlah_penduduk_miskin', 'garis_kemiskinan', 'jumlah_pengangguran', 'PDRB'],
                        index=3
                    )
                with col2:
                    feature_y = st.selectbox(
                        "Pilih fitur untuk sumbu Y:",
                        ['jumlah_penduduk_miskin', 'garis_kemiskinan', 'jumlah_pengangguran', 'PDRB'],
                        index=0
                    )
                
                if feature_x != feature_y:
                    fig = create_feature_scatter_plot(df, feature_x, feature_y, cluster_data)
                    st.pyplot(fig)
                    st.caption(f"Clustering berdasarkan {feature_x} vs {feature_y}")
                else:
                    st.warning("Silakan pilih fitur yang berbeda untuk sumbu X dan Y")
    
    # ===== HASIL CLUSTERING =====
    elif menu == "🎯 Hasil Clustering":
        st.title("🎯 ANALISIS HASIL CLUSTERING")
        st.markdown("*Analisis Mendalam Tiap Cluster*")
        
        # Tab untuk berbagai visualisasi
        tab1, tab2 = st.tabs(["📊 Perbandingan", "🔍 Detail Cluster"])
        
        with tab1:
            st.subheader("📊 Perbandingan Antar Cluster")
            fig = create_kmeans_scatter_plot(df, cluster_data)
            st.pyplot(fig)
            
            # Tambah metrics tambahan di bawah visualisasi
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
            
            # Tambah insight dan interpretasi
            st.subheader("💡 Interpretasi Visualisasi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Indikator Kualitas Clustering:**
                1. **Separasi antar Cluster**: Centroid yang berjauhan menunjukkan cluster yang berbeda
                2. **Kepadatan Cluster**: Titik yang berkumpul rapat menunjukkan homogenitas
                3. **Outlier**: Wilayah yang jauh dari centroid perlu analisis khusus
                4. **Overlap**: Area tumpang tindih menunjukkan kesamaan karakteristik
                """)
            
            with col2:
                st.markdown("""
                **Rekomendasi Kebijakan:**
                1. **Cluster 0 (Ekonomi Tinggi)**: Fokus pada pemerataan
                2. **Cluster 1 (Ekonomi Menengah)**: Penguatan infrastruktur
                3. **Cluster 2 (Perlu Perhatian)**: Intervensi komprehensif
                """)
            
            # Visualisasi tambahan
            st.subheader("📊 Perbandingan Numerik Cluster")
            st.pyplot(create_cluster_comparison(cluster_data))
            
            # Tambah visualisasi tambahan
            st.subheader("📊 Visualisasi Tambahan Berdasarkan Fitur")
            
            viz_option = st.selectbox(
                "Pilih visualisasi tambahan:",
                ["PDRB vs Penduduk Miskin per Cluster", "Pengangguran vs Garis Kemiskinan per Cluster"],
                key="viz_option_hasil"
            )
            
            if viz_option == "PDRB vs Penduduk Miskin per Cluster":
                fig2 = create_feature_scatter_plot(df, 'PDRB', 'jumlah_penduduk_miskin', cluster_data)
                st.pyplot(fig2)
                st.caption("Hubungan antara PDRB dan Jumlah Penduduk Miskin per Cluster")
            else:
                fig2 = create_feature_scatter_plot(df, 'jumlah_pengangguran', 'garis_kemiskinan', cluster_data)
                st.pyplot(fig2)
                st.caption("Hubungan antara Pengangguran dan Garis Kemiskinan per Cluster")
        
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
                
                # Karakteristik Cluster
                st.markdown("**📋 Karakteristik Utama:**")
                if i == 0:
                    st.write("• PDRB tertinggi di antara semua cluster")
                    st.write("• Jumlah penduduk miskin masih tinggi meski ekonomi kuat")
                    st.write("• Dominan wilayah urban dan industri")
                elif i == 1:
                    st.write("• PDRB menengah dengan stabilitas ekonomi")
                    st.write("• Jumlah penduduk miskin relatif rendah")
                    st.write("• Perlu penguatan untuk mencapai ekonomi tinggi")
                else:
                    st.write("• PDRB terendah dan perlu perhatian khusus")
                    st.write("• Tingkat pengangguran dan kemiskinan tinggi")
                    st.write("• Membutuhkan intervensi kebijakan intensif")
                
                # Wilayah dalam cluster
                st.write("📍 **Daftar Wilayah:**")
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
    
# ===== DATASET =====
    else:
        st.title("📋 DATASET LENGKAP")
        st.markdown("*Tabel Data dengan Filter dan Download*")
        
        
        # Data Table
        st.subheader("📋 Tabel Data Lengkap")
        
        # Konfigurasi kolom untuk formatting yang lebih baik
        column_config = {
            "Tahun": st.column_config.NumberColumn(
                format="%d",
                help="Tahun data"
            ),
            "kabupaten_kota": st.column_config.TextColumn(
                "Kabupaten/Kota",
                help="Nama wilayah"
            ),
            "jumlah_warga_jabar": st.column_config.NumberColumn(
                "Jumlah Penduduk",
                format="%,d",
                help="Total penduduk wilayah"
            ),
            "jumlah_penduduk_miskin": st.column_config.NumberColumn(
                "Penduduk Miskin",
                format="%,d",
                help="Jumlah penduduk miskin"
            ),
            "garis_kemiskinan": st.column_config.NumberColumn(
                "Garis Kemiskinan",
                format="Rp %,.0f",
                help="Rata-rata pengeluaran per kapita per bulan"
            ),
            "jumlah_pengangguran": st.column_config.NumberColumn(
                "Pengangguran",
                format="%,d",
                help="Jumlah pengangguran"
            ),
            "PDRB": st.column_config.NumberColumn(
                "PDRB",
                format="Rp %,.0f",
                help="Produk Domestic Regional Bruto"
            ),
            "Cluster_KMeans": st.column_config.NumberColumn(
                "Cluster K-Means",
                format="%d",
                help="Hasil clustering K-Means"
            )
        }
        
        # Tampilkan dataframe dengan formatting
        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )
        
        # Informasi tambahan tentang cluster
        with st.expander("ℹ️ Informasi Kategori Cluster"):
            kategori_cluster = {
                0: "Wilayah Ekonomi Tinggi",
                1: "Wilayah Ekonomi Menengah", 
                2: "Wilayah Perlu Perhatian"
            }
            
            for cluster_num, kategori in kategori_cluster.items():
                st.write(f"**Cluster {cluster_num}**: {kategori}")
                if cluster_num in filtered_df['Cluster_KMeans'].unique():
                    cluster_data = filtered_df[filtered_df['Cluster_KMeans'] == cluster_num]
                    st.write(f"  - Jumlah wilayah: {len(cluster_data)}")
                    st.write(f"  - PDRB rata-rata: Rp {cluster_data['PDRB'].mean():,.0f}")
                    st.write(f"  - Penduduk miskin rata-rata: {cluster_data['jumlah_penduduk_miskin'].mean():,.0f} jiwa")
                st.write("---")
        
        # Download section
        st.subheader("💾 Download Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download data terfilter
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data Terfilter (CSV)",
                data=csv,
                file_name=f"data_kemiskinan_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download data dengan filter yang telah diterapkan"
            )
        
        with col2:
            # Download seluruh data
            csv_full = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Seluruh Data (CSV)",
                data=csv_full,
                file_name=f"data_kemiskinan_full_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download seluruh data tanpa filter"
            )
        
        # Quick summary
        st.subheader("📈 Ringkasan Cepat")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.write("**Wilayah dengan PDRB Tertinggi:**")
            max_pdrb = filtered_df.loc[filtered_df['PDRB'].idxmax()]
            st.write(f"{max_pdrb['kabupaten_kota']}")
            st.write(f"Rp {max_pdrb['PDRB']:,.0f}")
            st.write(f"Cluster: {int(max_pdrb['Cluster_KMeans'])}")
        
        with summary_col2:
            st.write("**Wilayah dengan Penduduk Miskin Tertinggi:**")
            max_miskin = filtered_df.loc[filtered_df['jumlah_penduduk_miskin'].idxmax()]
            st.write(f"{max_miskin['kabupaten_kota']}")
            st.write(f"{max_miskin['jumlah_penduduk_miskin']:,} jiwa")
            st.write(f"Cluster: {int(max_miskin['Cluster_KMeans'])}")
        
        with summary_col3:
            st.write("**Distribusi Tahun:**")
            tahun_counts = filtered_df['Tahun'].value_counts().sort_index()
            for tahun, count in tahun_counts.items():
                st.write(f"Tahun {tahun}: {count} wilayah")

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()


