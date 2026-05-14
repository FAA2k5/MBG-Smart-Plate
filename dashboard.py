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

# ============================================================
# LOAD DATA (VERSI DARI FILE CSV)
# ============================================================
@st.cache_data
def load_bmi_data():
    """Memuat dataset antropometri BMI"""
    
    # Opsi 1: Load dari file CSV (jika sudah disimpan)
    if os.path.exists('df_bmi_clean.csv'):
        df = pd.read_csv('df_bmi_clean.csv')
        print(f"✅ Load dari file lokal: {df.shape}")
        return df
    
    # Opsi 2: Load dari URL (fallback)
    url_bmi = 'https://docs.google.com/spreadsheets/d/1ub4EZ50clYbgw_yi0HQX-2OD99cXmMa4-NxrgX3tn88/export?format=csv'
    df = pd.read_csv(url_bmi)
    
    # Rename columns
    df = df.rename(columns={
        'umur': 'Age',
        'jenis_kelamin': 'Gender',
        'tinggi_cm': 'Height',
        'berat_kg': 'Weight',
        'bmi': 'BMI',
        'label': 'Status_Gizi'
    })
    
    # Standardisasi Gender
    df['Gender'] = df['Gender'].replace({'Pria': 'Laki-laki', 'Wanita': 'Perempuan'})
    
    # Standardisasi Status_Gizi
    status_mapping = {
        'Kurus': 'Underweight', 'Underweight': 'Underweight',
        'Normal': 'Normal', 'Gemuk': 'Overweight',
        'Overweight': 'Overweight', 'Obesitas': 'Obesity', 'Obesity': 'Obesity'
    }
    df['Status_Gizi'] = df['Status_Gizi'].replace(status_mapping)
    
    # Encode Gender
    df['Gender_Encoded'] = df['Gender'].map({'Laki-laki': 1, 'Perempuan': 0})
    
    # Filter data valid
    valid_status = ['Underweight', 'Normal', 'Overweight', 'Obesity']
    df = df[df['Status_Gizi'].isin(valid_status)]
    df = df[(df['Age'] >= 5) & (df['Age'] <= 16)]
    df = df[(df['Height'] >= 80) & (df['Height'] <= 200)]
    df = df[(df['Weight'] >= 10) & (df['Weight'] <= 150)]
    df = df.dropna()
    
    # Hitung BMI
    df['BMI'] = round(df['Weight'] / ((df['Height']/100) ** 2), 2)
    
    print(f"✅ Load dari URL: {df.shape}")
    return df

@st.cache_data
def load_tkpi_data():
    """Memuat dataset TKPI"""
    url_tkpi = 'https://docs.google.com/spreadsheets/d/1VcYK__gUuteds8Hk7OT9S_tEoLCyRqy3/export?format=csv'
    df = pd.read_csv(url_tkpi)
    
    # Pilih kolom yang tersedia
    available_cols = ['NAMA BAHAN', 'ENERGI (Kal)', 'PROTEIN (g)', 'LEMAK (g)', 'KH (g)', 'SERAT (g)']
    existing_cols = [col for col in available_cols if col in df.columns]
    df = df[existing_cols].copy()
    
    # Bersihkan nama bahan
    df['NAMA BAHAN'] = df['NAMA BAHAN'].astype(str).str.replace('\n', ' ', regex=False).str.strip()
    
    # Konversi ke numeric
    for col in existing_cols[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Hapus missing dan energi <= 0
    df = df.dropna()
    df = df[df['ENERGI (Kal)'] > 0]
    
    # Filter bahan haram
    filter_keywords = 'babi|hiu|anjing|cacing'
    df = df[~df['NAMA BAHAN'].str.contains(filter_keywords, case=False, na=False)]
    
    return df

# ============================================================
# FUNGSI REKOMENDASI BAHAN
# ============================================================
def get_rekomendasi(kategori, df_tkpi):
    if kategori == 'Underweight':
        rekomendasi = df_tkpi.nlargest(8, 'ENERGI (Kal)')[['NAMA BAHAN', 'ENERGI (Kal)', 'PROTEIN (g)', 'LEMAK (g)', 'KH (g)']]
        target = "Meningkatkan berat badan"
        strategi = "Tingkatkan asupan kalori dan protein"
        warna = "#E74C3C"
    elif kategori in ['Overweight', 'Obesity']:
        rekomendasi = df_tkpi.nlargest(8, 'SERAT (g)')[['NAMA BAHAN', 'SERAT (g)', 'ENERGI (Kal)', 'PROTEIN (g)', 'KH (g)']]
        target = "Mengontrol/Menurunkan berat badan"
        strategi = "Kurangi kalori, perbanyak serat"
        warna = "#2ECC71"
    else:
        rekomendasi = df_tkpi.nlargest(8, 'PROTEIN (g)')[['NAMA BAHAN', 'PROTEIN (g)', 'ENERGI (Kal)', 'SERAT (g)', 'LEMAK (g)']]
        target = "Mempertahankan berat badan ideal"
        strategi = "Pola makan seimbang sesuai AKG"
        warna = "#3498DB"
    return rekomendasi, target, strategi, warna

# ============================================================
# FUNGSI TRAIN MODEL
# ============================================================
@st.cache_resource
def train_model(df_bmi):
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
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, le, accuracy, X_test_scaled, y_test, y_pred

# ============================================================
# FUNGSI PREDIKSI
# ============================================================
def predict_status(model, scaler, le, age, gender, height, weight):
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

# Main content
st.title("🍽️ MBG Smart Plate")
st.markdown("### Dashboard Sistem Personalisasi Menu Gizi Berbasis Web")

# Load data
with st.spinner("Memuat data..."):
    df_bmi = load_bmi_data()
    df_tkpi = load_tkpi_data()
    
    # Cek apakah data kosong
    if len(df_bmi) == 0:
        st.error("❌ Data BMI kosong! Periksa kembali sumber data.")
        st.stop()
    
    model, scaler, le, accuracy, X_test, y_test, y_pred = train_model(df_bmi)

st.success(f"✅ Data berhasil dimuat! Total {len(df_bmi)} siswa, {len(df_tkpi)} bahan pangan")

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
st.caption("Copyright © 2026 | MBG Smart Plate - Sistem Personalisasi Menu Gizi")
