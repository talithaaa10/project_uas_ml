import streamlit as st
import json
import pandas as pd
import joblib

st.set_page_config(page_title="Analisis Kemiskinan", layout="wide")

# TITLE
st.title("📊 ANALISIS CLUSTERING KEMISKINAN JAWA BARAT")
st.markdown("Proyek Machine Learning - Segmentasi Wilayah Berdasarkan Indikator Ekonomi")

# LOAD DATA
try:
    # Load cluster analysis
    with open('hasil_clustering.json', 'r') as f:
        cluster_data = json.load(f)
    
    # Load dataset
    df = pd.read_csv('datasetkemiskinan_final.csv')
    
    st.success("✅ Data berhasil dimuat")
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()

# DASHBOARD
st.header("📈 DASHBOARD")

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Akurasi Model", f"{cluster_data['skor_silhouette']:.3f}")
with col2:
    st.metric("Total Data", len(df))
with col3:
    st.metric("Periode", f"{df['Tahun'].min()}-{df['Tahun'].max()}")

# CLUSTER ANALYSIS
st.header("🎯 HASIL CLUSTERING")

for i in range(3):
    c = cluster_data['cluster'][str(i)]
    
    with st.expander(f"🔵 CLUSTER {i}: {c['kategori']} ({c['jumlah']} wilayah)"):
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("📊 Indikator Ekonomi:")
            st.write(f"• PDRB Rata-rata: *Rp {c['pdrb_rata']:,.0f}*")
            st.write(f"• Garis Kemiskinan: Rp {c['garis_kemiskinan_rata']:,.0f}")
        
        with col2:
            st.write("👥 Kondisi Sosial:")
            st.write(f"• Penduduk Miskin: *{c['miskin_rata']:.1f} ribu*")
            st.write(f"• Pengangguran: {c['pengangguran_rata']:,.0f} jiwa")
        
        st.write("📍 Contoh Wilayah:")
        for wilayah in c['contoh_wilayah']:
            st.write(f"- {wilayah}")

# PREDICTION FORM
st.header("🔮 PREDIKSI CLUSTER BARU")

with st.form("prediction_form"):
    st.write("Masukkan data wilayah untuk prediksi:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pdrb = st.number_input("PDRB", min_value=10000.0, value=50000.0, step=10000.0)
        penduduk = st.number_input("Jumlah Penduduk (ribu)", min_value=100.0, value=2000.0, step=100.0)
    
    with col2:
        miskin = st.number_input("Penduduk Miskin (ribu)", min_value=10.0, value=150.0, step=10.0)
        pengangguran = st.number_input("Pengangguran (jiwa)", min_value=1000, value=50000, step=1000)
    
    submit = st.form_submit_button("🚀 Prediksi")
    
    if submit:
        try:
            # Load model
            bundle = joblib.load('kmeans.pkl')
            model = bundle['model']
            scaler = bundle['scaler']
            
            # Prepare input (use average garis_kemiskinan from cluster 0 as default)
            garis_kemiskinan_default = 350000
            input_data = [[penduduk, miskin, garis_kemiskinan_default, pengangguran, pdrb]]
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            cluster_info = cluster_data['cluster'][str(prediction)]
            
            # Show result
            st.success(f"✅ *Hasil Prediksi: Cluster {prediction}*")
            st.info(f"*Kategori:* {cluster_info['kategori']}")
            st.write(f"*Karakteristik serupa dengan wilayah:* {', '.join(cluster_info['contoh_wilayah'])}")
            
        except Exception as e:
            st.error(f"Error dalam prediksi: {str(e)}")

# DATA TABLE
st.header("📋 DATA LENGKAP")
st.dataframe(df, use_container_width=True)

# FOOTER
st.divider()
st.caption("Proyek Akhir Machine Learning | Teknik Informatika | 2025")
