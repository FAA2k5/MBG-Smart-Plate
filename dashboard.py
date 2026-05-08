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
import joblib

warnings.filterwarnings('ignore')

# Set style
sns.set(style='whitegrid')
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_bmi_data():
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
    
    # Encode Gender
    df['Gender_Encoded'] = df['Gender'].map({'P': 1, 'L': 0})
    
    # Clean missing values
    df = df.dropna()
    
    return df

@st.cache_data
def load_tkpi_data():
    url_tkpi = 'https://docs.google.com/spreadsheets/d/1VcYK__gUuteds8Hk7OT9S_tEoLCyRqy3/export?format=csv'
    df = pd.read_csv(url_tkpi)
    
    # Clean column names and select relevant columns
    df = df[['NAMA BAHAN', 'ENERGI (Kal)', 'PROTEIN (g)', 'LEMAK (g)', 'KH (g)', 'SERAT (g)']].copy()
    
    # Bersihkan nama bahan dari karakter \n
    df['NAMA BAHAN'] = df['NAMA BAHAN'].astype(str).str.replace('\n', ' ', regex=False).str.strip()
    
    # Konversi ke numeric
    for col in ['ENERGI (Kal)', 'PROTEIN (g)', 'LEMAK (g)', 'KH (g)', 'SERAT (g)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Hapus missing values dan energi 0
    df = df.dropna()
    df = df[df['ENERGI (Kal)'] > 0]
    
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
        rekomendasi = df_tkpi.nlargest(8, 'SERAT (g)')[['NAMA BAHAN', 'ENERGI (Kal)', 'SERAT (g)', 'PROTEIN (g)', 'LEMAK (g)']]
        target = "Mengontrol/Menurunkan berat badan"
        strategi = "Kurangi kalori, perbanyak serat dan protein tanpa lemak"
        warna = "#2ECC71"
    else:  # Normal
        rekomendasi = df_tkpi.nlargest(8, 'PROTEIN (g)')[['NAMA BAHAN', 'PROTEIN (g)', 'ENERGI (Kal)', 'SERAT (g)', 'LEMAK (g)']]
        target = "Mempertahankan berat badan ideal"
        strategi = "Pola makan seimbang sesuai AKG"
        warna = "#3498DB"
    
    return rekomendasi, target, strategi, warna

# ============================================================
# FUNGSI TRAIN MODEL (Cached)
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
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, le, accuracy, X_test_scaled, y_test, y_pred

# ============================================================
# FUNGSI PREDIKSI INDIVIDU
# ============================================================
def predict_status(model, scaler, le, age, gender, height, weight):
    gender_enc = 1 if gender == 'Pria' else 0
    features = np.array([[age, height, weight, gender_enc]])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    status = le.inverse_transform([pred])[0]
    return status

# ============================================================
# DASHBOARD UTAMA
# ============================================================
st.set_page_config(page_title="MBG Smart Plate", page_icon="🍽️", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2838/2838912.png", width=80)
    st.title("MBG Smart Plate")
    st.markdown("### Sistem Personalisasi Menu Gizi")
    st.markdown("---")
    st.markdown("**Program Makan Bergizi Gratis (MBG)**")
    st.markdown("Kementerian Kesehatan RI")
    st.markdown("---")
    st.markdown("**Tim Pengembang:**")
    st.markdown("- Fairuz Arfan Abhipraya (DS)")
    st.markdown("- Andika Bagus Saputra (DS)")
    st.markdown("- Erlang Sinatrya (Fullstack)")
    st.markdown("- Ardhana Prasasta (Fullstack)")
    st.markdown("- Arya Budi Raharja (AI/ML)")
    st.markdown("- Marchelino Lumantow (AI/ML)")

# Main content
st.title("🍽️ MBG Smart Plate")
st.markdown("### Dashboard Sistem Personalisasi Menu Gizi Berbasis Web")
st.markdown("Mendukung Program Makan Bergizi Gratis (MBG)")

# Load data
with st.spinner("Memuat data..."):
    df_bmi = load_bmi_data()
    df_tkpi = load_tkpi_data()
    model, scaler, le, accuracy, X_test, y_test, y_pred = train_model(df_bmi)

st.success("✅ Data berhasil dimuat!")

# ============================================================
# METRIK UTAMA
# ============================================================
st.markdown("---")
st.subheader("📊 Ringkasan Data")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Data Siswa", f"{len(df_bmi):,}")

with col2:
    st.metric("Kategori Status Gizi", f"{df_bmi['Status_Gizi'].nunique()}")

with col3:
    st.metric("Akurasi Model", f"{accuracy:.2%}")

with col4:
    st.metric("Total Bahan Pangan", f"{len(df_tkpi):,}")

# ============================================================
# DISTRIBUSI STATUS GIZI
# ============================================================
st.markdown("---")
st.subheader("📈 Distribusi Status Gizi Siswa")

col1, col2 = st.columns(2)

with col1:
    status_counts = df_bmi['Status_Gizi'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_status = {'Normal': '#2ECC71', 'Kurus': '#E74C3C', 'Gemuk': '#F39C12', 'Obesitas': '#C0392B'}
    bar_colors = [colors_status.get(s, '#3498DB') for s in status_counts.index]
    
    bars = ax.bar(status_counts.index, status_counts.values, color=bar_colors)
    ax.set_title('Distribusi Status Gizi Siswa', fontsize=14)
    ax.set_xlabel('Status Gizi')
    ax.set_ylabel('Jumlah Siswa')
    
    for bar, val in zip(bars, status_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center')
    
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Pie chart untuk persentase
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
           colors=bar_colors, explode=[0.03] * len(status_counts))
    ax.set_title('Persentase Status Gizi', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# ANALISIS PER KELOMPOK USIA
# ============================================================
st.markdown("---")
st.subheader("👥 Analisis per Kelompok Usia")

# Buat kelompok usia
bins = [5, 8, 11, 14, 17]
labels = ['5-7 tahun', '8-10 tahun', '11-13 tahun', '14-16 tahun']
df_bmi['Age_Group'] = pd.cut(df_bmi['Age'], bins=bins, labels=labels, right=False)

# Crosstab
age_status = pd.crosstab(df_bmi['Age_Group'], df_bmi['Status_Gizi'], normalize='index') * 100

fig, ax = plt.subplots(figsize=(10, 6))
age_status.plot(kind='bar', stacked=True, ax=ax, color=[colors_status.get(s, '#3498DB') for s in age_status.columns])
ax.set_title('Distribusi Status Gizi per Kelompok Usia', fontsize=14)
ax.set_xlabel('Kelompok Usia')
ax.set_ylabel('Persentase (%)')
ax.legend(title='Status Gizi', bbox_to_anchor=(1.05, 1))
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig)

# ============================================================
# PREDIKSI INDIVIDU
# ============================================================
st.markdown("---")
st.subheader("🔍 Prediksi Status Gizi Individu")

col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input("Usia (tahun)", min_value=5, max_value=16, value=10, step=1)

with col2:
    gender = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])

with col3:
    height = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=180.0, value=130.0, step=0.1)

with col4:
    weight = st.number_input("Berat Badan (kg)", min_value=10.0, max_value=100.0, value=30.0, step=0.1)

if st.button("🔮 Prediksi Status Gizi", type="primary"):
    status = predict_status(model, scaler, le, age, gender, height, weight)
    
    # Tampilkan hasil prediksi
    if status == 'Normal':
        st.success(f"✅ **Hasil Prediksi: {status}**")
        st.info("💡 Saran: Pertahankan pola makan seimbang dan aktifitas fisik teratur.")
    elif status == 'Kurus':
        st.warning(f"⚠️ **Hasil Prediksi: {status}**")
        st.info("💡 Saran: Tingkatkan asupan kalori, protein, dan karbohidrat. Konsultasi dengan ahli gizi.")
    else:
        st.error(f"⚠️ **Hasil Prediksi: {status}**")
        st.info("💡 Saran: Kurangi asupan kalori dan lemak, perbanyak serat dan aktifitas fisik.")

# ============================================================
# REKOMENDASI BAHAN PANGAN
# ============================================================
st.markdown("---")
st.subheader("🥗 Rekomendasi Bahan Pangan Berdasarkan Status Gizi")

tab1, tab2, tab3 = st.tabs(["🍽️ Underweight (Kurus)", "⚖️ Normal", "🥬 Overweight/Obesitas (Gemuk/Obesitas)"])

with tab1:
    st.markdown("#### Rekomendasi untuk Kategori Underweight")
    st.markdown("**Target:** Meningkatkan berat badan")
    st.markdown("**Strategi:** Tingkatkan asupan kalori dan protein")
    
    rekomendasi, _, _, _ = get_rekomendasi('Underweight', df_tkpi)
    st.dataframe(rekomendasi, use_container_width=True)

with tab2:
    st.markdown("#### Rekomendasi untuk Kategori Normal")
    st.markdown("**Target:** Mempertahankan berat badan ideal")
    st.markdown("**Strategi:** Pola makan seimbang sesuai AKG")
    
    rekomendasi, _, _, _ = get_rekomendasi('Normal', df_tkpi)
    st.dataframe(rekomendasi, use_container_width=True)

with tab3:
    st.markdown("#### Rekomendasi untuk Kategori Overweight/Obesitas")
    st.markdown("**Target:** Mengontrol/Menurunkan berat badan")
    st.markdown("**Strategi:** Kurangi kalori, perbanyak serat dan protein tanpa lemak")
    
    rekomendasi, _, _, _ = get_rekomendasi('Overweight', df_tkpi)
    st.dataframe(rekomendasi, use_container_width=True)

# ============================================================
# VISUALISASI BAHAN PANGAN
# ============================================================
st.markdown("---")
st.subheader("📊 Visualisasi Bahan Pangan")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔋 10 Bahan dengan Energi Tertinggi")
    top_energy = df_tkpi.nlargest(10, 'ENERGI (Kal)')[['NAMA BAHAN', 'ENERGI (Kal)', 'PROTEIN (g)']]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_energy, x='ENERGI (Kal)', y='NAMA BAHAN', ax=ax, palette='Reds_r')
    ax.set_title('10 Bahan dengan Energi Tertinggi', fontsize=12)
    ax.set_xlabel('Energi (Kalori)')
    ax.set_ylabel('Nama Bahan')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("#### 🌿 10 Bahan dengan Serat Tertinggi")
    top_fiber = df_tkpi.nlargest(10, 'SERAT (g)')[['NAMA BAHAN', 'SERAT (g)', 'ENERGI (Kal)']]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_fiber, x='SERAT (g)', y='NAMA BAHAN', ax=ax, palette='Greens_r')
    ax.set_title('10 Bahan dengan Serat Tertinggi', fontsize=12)
    ax.set_xlabel('Serat (gram)')
    ax.set_ylabel('Nama Bahan')
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# EVALUASI MODEL
# ============================================================
st.markdown("---")
st.subheader("📋 Evaluasi Model Klasifikasi")

col1, col2 = st.columns(2)

with col1:
    st.metric("Akurasi Model Random Forest", f"{accuracy:.2%}")

with col2:
    # Classification Report
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)

# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax.set_title('Confusion Matrix - Random Forest', fontsize=14)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
st.pyplot(fig)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
st.markdown("---")
st.subheader("📊 Feature Importance")

features = ['Usia', 'Tinggi Badan', 'Berat Badan', 'Jenis Kelamin']
importances = model.feature_importances_

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=importances, y=features, ax=ax, palette='viridis')
ax.set_title('Feature Importance - Faktor Penentu Status Gizi', fontsize=14)
ax.set_xlabel('Nilai Importance')
ax.set_ylabel('Fitur')

for i, (imp, feat) in enumerate(zip(importances, features)):
    ax.text(imp + 0.01, i, f'{imp:.2%}', va='center')

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.caption("Copyright © 2026 | MBG Smart Plate - Sistem Personalisasi Menu Gizi")
