import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import os

warnings.filterwarnings('ignore')

# Set style
sns.set(style='whitegrid')
plt.style.use('seaborn-v0_8-darkgrid')

@st.cache_data
def load_bmi_data():
    """Memuat dataset antropometri BMI"""
    
    # Opsi 1: Coba load dari file lokal 
    if os.path.exists('df_bmi_clean.csv'):
        df = pd.read_csv('df_bmi_clean.csv')
        st.success(f"✅ Load dari file lokal: {df.shape[0]} siswa")
        return df
    
    # Opsi 2: Load dari URL Google Sheets
    # Menggunakan URL yang sama seperti di notebook Anda
    url_bmi = 'https://docs.google.com/spreadsheets/d/1E-sdZCDNeM-6gXMc72xx4si10InHdp5DKERisgkzDSs/export?format=csv'
    
    try:
        df = pd.read_csv(url_bmi)
        st.success(f"✅ Load dari URL: {df.shape[0]} siswa")
        
        # Hitung BMI (sama seperti di notebook)
        df['BMI'] = round(df['Weight'] / ((df['Height'] / 100) ** 2), 2)
        
        return df
    except Exception as e:
        st.error(f"❌ Gagal memuat data BMI: {e}")
        st.stop()

@st.cache_data
def load_tkpi_data():
    """Memuat dataset TKPI"""
    
    # Opsi 1: Coba load dari file lokal
    if os.path.exists('df_tkpi_clean.csv'):
        df = pd.read_csv('df_tkpi_clean.csv')
        st.success(f"✅ Load dari file lokal: {df.shape[0]} bahan")
        return df
    
    # Opsi 2: Load dari URL Google Sheets
    url_tkpi = 'https://docs.google.com/spreadsheets/d/1GZK2vG-8LBrHfDwedrBRZ6AnnrCYrgJwdaqJBgduFJc/export?format=csv'
    
    try:
        df = pd.read_csv(url_tkpi)
        st.success(f"✅ Load dari URL: {df.shape[0]} bahan")
        return df
    except Exception as e:
        st.error(f"❌ Gagal memuat data TKPI: {e}")
        st.stop()

# ============================================================
# FUNGSI REKOMENDASI BAHAN
# ============================================================
def get_rekomendasi(kategori, df_tkpi):
    """
    Memberikan rekomendasi bahan pangan berdasarkan kategori status gizi
    """
    # Filter bahan dengan energi > 0
    df_tkpi_filtered = df_tkpi[df_tkpi['ENERGI (Kal)'] > 0].copy()
    
    if kategori == 'Underweight':
        # Prioritas: Energi Tinggi, Protein Tinggi
        rekomendasi = df_tkpi_filtered.nlargest(8, 'ENERGI (Kal)')[
            ['NAMA BAHAN', 'ENERGI (Kal)', 'PROTEIN (g)', 'LEMAK (g)', 'KH (g)']
        ]
        target = "Meningkatkan berat badan"
        strategi = "Tingkatkan asupan kalori dan protein"
        warna = "#E74C3C"
        
    elif kategori in ['Overweight', 'Obesity']:
        # Prioritas: Serat Tinggi, Energi Rendah
        rekomendasi = df_tkpi_filtered.nlargest(8, 'SERAT (g)')[
            ['NAMA BAHAN', 'SERAT (g)', 'ENERGI (Kal)', 'PROTEIN (g)', 'KH (g)']
        ]
        target = "Mengontrol/Menurunkan berat badan"
        strategi = "Kurangi kalori, perbanyak serat"
        warna = "#2ECC71"
        
    else:  # Normal
        # Prioritas: Gizi Seimbang, Protein cukup
        rekomendasi = df_tkpi_filtered.nlargest(8, 'PROTEIN (g)')[
            ['NAMA BAHAN', 'PROTEIN (g)', 'ENERGI (Kal)', 'SERAT (g)', 'LEMAK (g)']
        ]
        target = "Mempertahankan berat badan ideal"
        strategi = "Pola makan seimbang sesuai AKG"
        warna = "#3498DB"
    
    return rekomendasi, target, strategi, warna

# ============================================================
# FUNGSI TRAIN MODEL
# ============================================================
@st.cache_resource
def train_model(df_bmi):
    """
    Melatih model Random Forest untuk klasifikasi status gizi
    """
    features = ['Age', 'Height', 'Weight', 'Gender_Encoded']
    X = df_bmi[features]
    y = df_bmi['Status_Gizi']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, le, accuracy, X_test_scaled, y_test, y_pred

# ============================================================
# FUNGSI PREDIKSI
# ============================================================
def predict_status(model, scaler, le, age, gender, height, weight):
    """
    Memprediksi status gizi berdasarkan input pengguna
    """
    gender_enc = 1 if gender == 'Laki-laki' else 0
    features = np.array([[age, height, weight, gender_enc]])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    status = le.inverse_transform([pred])[0]
    return status

# ============================================================
# MAIN DASHBOARD
# ============================================================
st.set_page_config(page_title="MBG Smart Plate", page_icon="🍽️", layout="wide")

# Load data
with st.spinner("Memuat data..."):
    df_bmi = load_bmi_data()
    df_tkpi = load_tkpi_data()

# Train model
with st.spinner("Melatih model prediksi..."):
    model, scaler, le, accuracy, X_test, y_test, y_pred = train_model(df_bmi)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2838/2838912.png", width=80)
    st.title("MBG Smart Plate")
    st.markdown("### Sistem Personalisasi Menu Gizi")
    st.markdown("---")
    st.markdown("**Program Makan Bergizi Gratis (MBG)**")
    st.markdown("Badan Gizi Nasional")
    st.markdown("---")
    st.markdown("**Tim Pengembang:**")
    st.markdown("- Andika Bagus Saputra (Data Scientist)")
    st.markdown("- Fairuz Arfan Abhipraya (Data Scientist)")
    st.markdown("- Erlang Sinatrya (Full-stack Developer)")
    st.markdown("- Ardhana Prasasta (Full-stack Developer)")
    st.markdown("- Arya Budi Raharja (AI Engineer)")
    st.markdown("- Marchelino Lumantow (AI Engineer)")
    st.markdown("---")
    st.markdown("### 📊 Informasi Data")
    st.markdown(f"- **Total Siswa:** {len(df_bmi):,}")
    st.markdown(f"- **Total Bahan:** {len(df_tkpi):,}")
    st.markdown(f"- **Akurasi Model:** {accuracy:.2%}")

# Main content
st.title("🍽️ MBG Smart Plate")
st.markdown("### Dashboard Sistem Personalisasi Menu Gizi Berbasis Web")

# ============================================================
# METRIK UTAMA
# ============================================================
st.markdown("---")
st.subheader("📊 Ringkasan Data")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Data Siswa", f"{len(df_bmi):,}")

with col2:
    st.metric("Kategori Status Gizi", f"{df_bmi['Status_Gizi'].nunique()}")

with col3:
    st.metric("Akurasi Model", f"{accuracy:.2%}")

with col4:
    st.metric("Total Bahan Pangan", f"{len(df_tkpi):,}")

with col5:
    prioritas = df_bmi[df_bmi['Status_Gizi'].isin(['Underweight', 'Overweight', 'Obesity'])].shape[0]
    st.metric("Siswa Prioritas", f"{prioritas:,}")

st.markdown("---")

# ============================================================
# TAB 1: DISTRIBUSI STATUS GIZI
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribusi Status Gizi", "🎯 Prediksi Status Gizi", "🍎 Rekomendasi Menu", "📊 Model Performance"])

with tab1:
    st.subheader("Distribusi Status Gizi Siswa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        status_counts = df_bmi['Status_Gizi'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = {
            'Underweight': '#E74C3C',
            'Normal': '#2ECC71',
            'Overweight': '#F39C12',
            'Obesity': '#C0392B'
        }
        pie_colors = [colors.get(s, '#3498DB') for s in status_counts.index]
        
        wedges, texts, autotexts = ax.pie(
            status_counts.values,
            labels=status_counts.index,
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=90,
            explode=[0.03] * len(status_counts)
        )
        ax.set_title('Distribusi Status Gizi', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(status_counts.index, status_counts.values, 
                      color=[colors.get(s, '#3498DB') for s in status_counts.index],
                      edgecolor='black')
        ax.set_title('Jumlah Siswa per Status Gizi', fontsize=14, fontweight='bold')
        ax.set_xlabel('Status Gizi', fontsize=12)
        ax.set_ylabel('Jumlah Siswa', fontsize=12)
        
        for bar, count in zip(bars, status_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   str(count), ha='center', fontweight='bold')
        
        st.pyplot(fig)
        plt.close(fig)
    
    # Tabel distribusi
    st.subheader("Detail Distribusi")
    status_df = df_bmi['Status_Gizi'].value_counts().reset_index()
    status_df.columns = ['Status Gizi', 'Jumlah Siswa']
    status_df['Persentase'] = (status_df['Jumlah Siswa'] / len(df_bmi) * 100).round(1).astype(str) + '%'
    st.dataframe(status_df, use_container_width=True)

with tab2:
    st.subheader("🎯 Prediksi Status Gizi")
    st.markdown("Masukkan data siswa untuk memprediksi status gizi dan mendapatkan rekomendasi menu.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Usia (tahun)", min_value=5, max_value=16, value=10, step=1)
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    
    with col2:
        height = st.number_input("Tinggi Badan (cm)", min_value=80.0, max_value=200.0, value=130.0, step=1.0)
        weight = st.number_input("Berat Badan (kg)", min_value=10.0, max_value=150.0, value=30.0, step=1.0)
    
    if st.button("🔍 Prediksi Status Gizi", type="primary", use_container_width=True):
        with st.spinner("Memproses prediksi..."):
            status = predict_status(model, scaler, le, age, gender, height, weight)
            
            # Tampilkan hasil prediksi
            st.markdown("---")
            st.subheader("📋 Hasil Prediksi")
            
            # Warna berdasarkan status
            status_colors = {
                'Underweight': '#E74C3C',
                'Normal': '#2ECC71',
                'Overweight': '#F39C12',
                'Obesity': '#C0392B'
            }
            color = status_colors.get(status, '#3498DB')
            
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {color};">
                <h3 style="color: {color}; margin: 0;">Status Gizi: {status}</h3>
                <p style="margin-top: 10px; margin-bottom: 0;">
                    <strong>Usia:</strong> {age} tahun | 
                    <strong>Jenis Kelamin:</strong> {gender} | 
                    <strong>Tinggi:</strong> {height} cm | 
                    <strong>Berat:</strong> {weight} kg
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Tampilkan rekomendasi
            st.markdown("---")
            st.subheader("🍽️ Rekomendasi Menu")
            
            rekomendasi, target, strategi, warna_rek = get_rekomendasi(status, df_tkpi)
            
            st.markdown(f"""
            <div style="background-color: {warna_rek}20; padding: 15px; border-radius: 10px;">
                <p><strong>🎯 Target:</strong> {target}</p>
                <p><strong>📌 Strategi:</strong> {strategi}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(rekomendasi, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("🍎 Rekomendasi Menu Berdasarkan Status Gizi")
    
    # Pilih kategori
    kategori_options = ['Underweight', 'Normal', 'Overweight', 'Obesity']
    selected_kategori = st.selectbox("Pilih Kategori Status Gizi", kategori_options)
    
    rekomendasi, target, strategi, warna = get_rekomendasi(selected_kategori, df_tkpi)
    
    st.markdown(f"""
    <div style="background-color: {warna}20; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <p><strong>🎯 Target:</strong> {target}</p>
        <p><strong>📌 Strategi:</strong> {strategi}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(rekomendasi, use_container_width=True, hide_index=True)
    
    # Visualisasi rekomendasi
    st.subheader("📊 Visualisasi Rekomendasi")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if selected_kategori == 'Underweight':
        sns.barplot(data=rekomendasi, y='NAMA BAHAN', x='ENERGI (Kal)', 
                   palette='Reds_r', ax=ax)
        ax.set_title(f'Rekomendasi Bahan untuk {selected_kategori}: Energi Tertinggi', fontsize=12)
        ax.set_xlabel('Energi (Kalori)', fontsize=10)
        
    elif selected_kategori in ['Overweight', 'Obesity']:
        sns.barplot(data=rekomendasi, y='NAMA BAHAN', x='SERAT (g)', 
                   palette='Greens_r', ax=ax)
        ax.set_title(f'Rekomendasi Bahan untuk {selected_kategori}: Serat Tertinggi', fontsize=12)
        ax.set_xlabel('Serat (gram)', fontsize=10)
        
    else:
        sns.barplot(data=rekomendasi, y='NAMA BAHAN', x='PROTEIN (g)', 
                   palette='Blues_r', ax=ax)
        ax.set_title(f'Rekomendasi Bahan untuk {selected_kategori}: Protein Tertinggi', fontsize=12)
        ax.set_xlabel('Protein (gram)', fontsize=10)
    
    ax.set_ylabel('Nama Bahan', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab4:
    st.subheader("📊 Performa Model Klasifikasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Akurasi Model", f"{accuracy:.2%}", delta=None)
        
        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)
    
    with col2:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        st.pyplot(fig)
        plt.close(fig)
    
    # Feature Importance
    st.subheader("Feature Importance (Pengaruh terhadap Status Gizi)")
    
    feature_names = ['Usia', 'Tinggi Badan', 'Berat Badan', 'Jenis Kelamin']
    importances = model.feature_importances_
    
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    
    bars = ax.bar([feature_names[i] for i in indices], importances[indices], color=colors, edgecolor='black')
    ax.set_title('Feature Importance - Faktor Penentu Status Gizi', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fitur', fontsize=12)
    ax.set_ylabel('Nilai Importance', fontsize=12)
    
    for bar, imp in zip(bars, importances[indices]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{imp:.3f}', ha='center', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("© 2026 MBG Smart Plate - Sistem Personalisasi Menu Gizi Berbasis Web | Badan Gizi Nasional")