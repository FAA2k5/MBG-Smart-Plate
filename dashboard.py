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
    """Memuat dataset antropometri BMI"""
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
    
    # Standardisasi Status_Gizi ke 4 kategori
    status_mapping = {
        'Kurus': 'Underweight',
        'Underweight': 'Underweight',
        'Normal': 'Normal',
        'Gemuk': 'Overweight',
        'Overweight': 'Overweight',
        'Obesitas': 'Obesity',
        'Obesity': 'Obesity'
    }
    df['Status_Gizi'] = df['Status_Gizi'].replace(status_mapping)
    
    # Encode Gender (Pria=1, Wanita=0)
    df['Gender_Encoded'] = df['Gender'].map({'Pria': 1, 'Wanita': 0})
    
    # Filter data valid (Age 6-12 tahun untuk SD)
    df = df[(df['Age'] >= 6) & (df['Age'] <= 12)]
    
    # Clean missing values
    df = df.dropna()
    
    return df

@st.cache_data
def load_tkpi_data():
    """Memuat dataset TKPI (Tabel Komposisi Pangan Indonesia)"""
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
    
    # Hapus missing values dan energi <= 0
    df = df.dropna()
    df = df[df['ENERGI (Kal)'] > 0]
    
    # Filter bahan haram (babi, hiu, dll)
    filter_keywords = 'babi|hiu|anjing|cacing|kucing'
    df = df[~df['NAMA BAHAN'].str.contains(filter_keywords, case=False, na=False)]
    
    return df

# ============================================================
# FUNGSI REKOMENDASI BAHAN
# ============================================================
def get_rekomendasi(kategori, df_tkpi):
    """
    Memberikan rekomendasi bahan pangan berdasarkan kategori status gizi
    """
    if kategori == 'Underweight':
        # Underweight: prioritas energi dan protein tinggi
        rekomendasi = df_tkpi.nlargest(8, 'ENERGI (Kal)')[['NAMA BAHAN', 'ENERGI (Kal)', 'PROTEIN (g)', 'LEMAK (g)', 'KH (g)']]
        target = "Meningkatkan berat badan"
        strategi = "Tingkatkan asupan kalori, protein, dan karbohidrat"
        warna = "#E74C3C"
        
    elif kategori == 'Normal':
        # Normal: prioritas protein cukup, gizi seimbang
        rekomendasi = df_tkpi.nlargest(8, 'PROTEIN (g)')[['NAMA BAHAN', 'PROTEIN (g)', 'ENERGI (Kal)', 'SERAT (g)', 'LEMAK (g)']]
        target = "Mempertahankan berat badan ideal"
        strategi = "Pola makan seimbang sesuai AKG"
        warna = "#2ECC71"
        
    elif kategori in ['Overweight', 'Obesity']:
        # Overweight/Obesity: prioritas serat tinggi, energi rendah
        rekomendasi = df_tkpi.nlargest(8, 'SERAT (g)')[['NAMA BAHAN', 'SERAT (g)', 'ENERGI (Kal)', 'PROTEIN (g)', 'KH (g)']]
        target = "Mengontrol/Menurunkan berat badan"
        strategi = "Kurangi kalori dan lemak, perbanyak serat"
        warna = "#3498DB"
    else:
        rekomendasi = df_tkpi.head(8)
        target = "Konsultasi dengan ahli gizi"
        strategi = "Evaluasi lebih lanjut"
        warna = "#95A5A6"
    
    return rekomendasi, target, strategi, warna

# ============================================================
# FUNGSI TRAIN MODEL (Cached)
# ============================================================
@st.cache_resource
def train_model(df_bmi):
    """Melatih model Random Forest untuk klasifikasi status gizi"""
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
# FUNGSI PREDIKSI INDIVIDU
# ============================================================
def predict_status(model, scaler, le, age, gender, height, weight):
    """Prediksi status gizi berdasarkan input individu"""
    gender_enc = 1 if gender == 'Laki-laki' else 0
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
    st.caption("v2.0 | MBG Smart Plate")

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
    # Hitung siswa prioritas intervensi
    prioritas = df_bmi[df_bmi['Status_Gizi'].isin(['Underweight', 'Overweight', 'Obesity'])].shape[0]
    st.metric("Siswa Prioritas Intervensi", f"{prioritas:,}")

# ============================================================
# DISTRIBUSI STATUS GIZI
# ============================================================
st.markdown("---")
st.subheader("📈 Distribusi Status Gizi Siswa")

col1, col2 = st.columns(2)

with col1:
    status_counts = df_bmi['Status_Gizi'].value_counts()
    
    # Warna untuk setiap status
    colors_status = {
        'Normal': '#2ECC71', 
        'Underweight': '#E74C3C', 
        'Overweight': '#F39C12', 
        'Obesity': '#C0392B'
    }
    bar_colors = [colors_status.get(s, '#3498DB') for s in status_counts.index]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(status_counts.index, status_counts.values, color=bar_colors, edgecolor='black')
    ax.set_title('Distribusi Status Gizi Siswa', fontsize=14)
    ax.set_xlabel('Status Gizi', fontsize=11)
    ax.set_ylabel('Jumlah Siswa', fontsize=11)
    
    for bar, val in zip(bars, status_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        status_counts.values, 
        labels=status_counts.index, 
        autopct='%1.1f%%', 
        colors=bar_colors,
        explode=[0.03] * len(status_counts),
        startangle=90
    )
    ax.set_title('Persentase Status Gizi', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# ANALISIS PER KELOMPOK USIA (QUANTILE)
# ============================================================
st.markdown("---")
st.subheader("👥 Analisis per Kelompok Usia (Quantile)")

# Buat kelompok usia dengan quantile
df_bmi['Age_Group'] = pd.qcut(df_bmi['Age'], q=4, labels=['Q1 (6-7 thn)', 'Q2 (8-9 thn)', 'Q3 (10-11 thn)', 'Q4 (12 thn)'])

# Crosstab
age_status = pd.crosstab(df_bmi['Age_Group'], df_bmi['Status_Gizi'], normalize='index') * 100

fig, ax = plt.subplots(figsize=(10, 6))
age_status.plot(kind='bar', stacked=True, ax=ax, color=[colors_status.get(s, '#3498DB') for s in age_status.columns], edgecolor='black')
ax.set_title('Distribusi Status Gizi per Kelompok Usia', fontsize=14)
ax.set_xlabel('Kelompok Usia', fontsize=11)
ax.set_ylabel('Persentase (%)', fontsize=11)
ax.legend(title='Status Gizi', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# ============================================================
# PREDIKSI INDIVIDU
# ============================================================
st.markdown("---")
st.subheader("🔍 Prediksi Status Gizi Individu")
st.markdown("Masukkan data antropometri siswa untuk mendapatkan klasifikasi status gizi dan rekomendasi menu.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input("Usia (tahun)", min_value=6, max_value=12, value=10, step=1)

with col2:
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

with col3:
    height = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=165.0, value=130.0, step=0.1)

with col4:
    weight = st.number_input("Berat Badan (kg)", min_value=15.0, max_value=65.0, value=30.0, step=0.1)

if st.button("🔮 Prediksi Status Gizi", type="primary", use_container_width=True):
    status = predict_status(model, scaler, le, age, gender, height, weight)
    
    # Tampilkan hasil prediksi dengan warna sesuai status
    if status == 'Normal':
        st.success(f"✅ **Hasil Prediksi: {status}**")
        st.info("💡 **Saran:** Pertahankan pola makan seimbang dengan gizi lengkap (karbohidrat, protein, lemak, vitamin, mineral). Lakukan aktivitas fisik teratur.")
    elif status == 'Underweight':
        st.warning(f"⚠️ **Hasil Prediksi: {status}**")
        st.info("💡 **Saran:** Tingkatkan asupan kalori, protein, dan karbohidrat. Konsumsi makanan padat energi seperti kacang-kacangan, telur, susu, dan daging. Konsultasi dengan ahli gizi.")
    elif status == 'Overweight':
        st.warning(f"⚠️ **Hasil Prediksi: {status}**")
        st.info("💡 **Saran:** Kurangi asupan kalori dan lemak jenuh. Perbanyak konsumsi serat (sayur dan buah). Tingkatkan aktivitas fisik.")
    else:  # Obesity
        st.error(f"⚠️ **Hasil Prediksi: {status}**")
        st.info("💡 **Saran:** Segera konsultasi dengan ahli gizi. Atur pola makan dengan kalori terkontrol, perbanyak serat, hindari gula berlebih, dan tingkatkan aktivitas fisik.")

# ============================================================
# REKOMENDASI BAHAN PANGAN (BERDASARKAN TKPI)
# ============================================================
st.markdown("---")
st.subheader("🥗 Rekomendasi Bahan Pangan Berdasarkan Status Gizi")

tab1, tab2, tab3 = st.tabs(["🔺 Underweight (Menaikkan BB)", "🟢 Normal (Menjaga BB)", "🔻 Overweight (Menurunkan BB)"])

with tab1:
    st.markdown("#### Rekomendasi untuk Kategori Underweight")
    st.markdown("**Target:** Meningkatkan berat badan")
    st.markdown("**Strategi:** Tingkatkan asupan kalori, protein, dan karbohidrat")
    
    rekomendasi, _, _, _ = get_rekomendasi('Underweight', df_tkpi)
    st.dataframe(rekomendasi, use_container_width=True)
    st.caption("Note: Bahan dengan energi tinggi cocok untuk menambah berat badan")

with tab2:
    st.markdown("#### Rekomendasi untuk Kategori Normal")
    st.markdown("**Target:** Mempertahankan berat badan ideal")
    st.markdown("**Strategi:** Pola makan seimbang dengan protein cukup")
    
    rekomendasi, _, _, _ = get_rekomendasi('Normal', df_tkpi)
    st.dataframe(rekomendasi, use_container_width=True)
    st.caption("Note: Bahan dengan protein tinggi untuk pertumbuhan optimal")

with tab3:
    st.markdown("#### Rekomendasi untuk Kategori Overweight/Obesitas")
    st.markdown("**Target:** Mengontrol/Menurunkan berat badan")
    st.markdown("**Strategi:** Kurangi kalori, perbanyak serat")
    
    rekomendasi, _, _, _ = get_rekomendasi('Overweight', df_tkpi)
    st.dataframe(rekomendasi, use_container_width=True)
    st.caption("Note: Bahan dengan serat tinggi memberikan rasa kenyang lebih lama")

# ============================================================
# VISUALISASI BAHAN PANGAN
# ============================================================
st.markdown("---")
st.subheader("📊 Analisis Bahan Pangan (TKPI)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔋 10 Bahan dengan Energi Tertinggi")
    top_energy = df_tkpi.nlargest(10, 'ENERGI (Kal)')[['NAMA BAHAN', 'ENERGI (Kal)', 'PROTEIN (g)', 'LEMAK (g)']]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_energy, x='ENERGI (Kal)', y='NAMA BAHAN', ax=ax, palette='Reds_r')
    ax.set_title('10 Bahan Pangan dengan Energi Tertinggi', fontsize=12)
    ax.set_xlabel('Energi (Kalori)', fontsize=10)
    ax.set_ylabel('Nama Bahan', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.dataframe(top_energy, use_container_width=True)

with col2:
    st.markdown("#### 🌿 10 Bahan dengan Serat Tertinggi")
    top_fiber = df_tkpi.nlargest(10, 'SERAT (g)')[['NAMA BAHAN', 'SERAT (g)', 'ENERGI (Kal)', 'PROTEIN (g)']]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_fiber, x='SERAT (g)', y='NAMA BAHAN', ax=ax, palette='Greens_r')
    ax.set_title('10 Bahan Pangan dengan Serat Tertinggi', fontsize=12)
    ax.set_xlabel('Serat (gram)', fontsize=10)
    ax.set_ylabel('Nama Bahan', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.dataframe(top_fiber, use_container_width=True)

# ============================================================
# EVALUASI MODEL KLASIFIKASI
# ============================================================
st.markdown("---")
st.subheader("📋 Evaluasi Model Klasifikasi")

col1, col2 = st.columns(2)

with col1:
    st.metric("Akurasi Model Random Forest", f"{accuracy:.2%}")
    st.metric("Jumlah Data Training", f"{int(len(df_bmi) * 0.8)} sampel")
    st.metric("Jumlah Data Testing", f"{int(len(df_bmi) * 0.2)} sampel")

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
ax.set_title('Confusion Matrix - Random Forest Classifier', fontsize=14)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label', fontsize=11)
plt.tight_layout()
st.pyplot(fig)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
st.markdown("---")
st.subheader("📊 Feature Importance - Faktor Penentu Status Gizi")

features = ['Usia (Age)', 'Tinggi Badan (Height)', 'Berat Badan (Weight)', 'Jenis Kelamin (Gender)']
importances = model.feature_importances_

# Urutkan
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(8, 5))
sorted_importances = importances[indices]
sorted_features = [features[i] for i in indices]
colors = ['#E74C3C' if f == 'Berat Badan (Weight)' else '#3498DB' for f in sorted_features]

bars = ax.barh(sorted_features, sorted_importances, color=colors, edgecolor='black')
ax.set_title('Feature Importance - Faktor Penentu Status Gizi', fontsize=14)
ax.set_xlabel('Nilai Importance', fontsize=11)
ax.set_ylabel('Fitur', fontsize=11)

for bar, imp in zip(bars, sorted_importances):
    ax.text(imp + 0.01, bar.get_y() + bar.get_height()/2, f'{imp:.2%}', va='center', fontsize=10)

plt.tight_layout()
st.pyplot(fig)

st.info("""
**Kesimpulan Feature Importance:**
- **Berat Badan (Weight)** adalah faktor paling dominan (paling tinggi pengaruhnya)
- **Tinggi Badan (Height)** berada di posisi kedua
- **Usia (Age)** dan **Jenis Kelamin (Gender)** memiliki pengaruh relatif kecil
- Untuk akurasi data lapangan, petugas SPPG harus memprioritaskan pengukuran berat badan yang akurat
""")

# ============================================================
# KESIMPULAN AKHIR
# ============================================================
st.markdown("---")
st.subheader("📝 Kesimpulan dan Rekomendasi")

with st.expander("📊 Kesimpulan Analisis Data"):
    st.markdown("""
    **1. Distribusi Status Gizi:**
    - Mayoritas siswa berada pada kategori **Normal**
    - Siswa yang memerlukan intervensi prioritas (Underweight + Overweight + Obesity) mencapai **{:.1f}%**
    
    **2. Parameter Gizi Prioritas:**
    - **Energi (Kalori)** dan **Protein** adalah parameter paling penting
    - **Serat** penting untuk kontrol berat badan
    - **Karbohidrat** sebagai sumber energi utama
    
    **3. Model Klasifikasi:**
    - **Random Forest** memberikan akurasi terbaik yaitu **{:.2%}**
    - Feature importance: Berat Badan > Tinggi Badan > Usia > Jenis Kelamin
    
    **4. Rekomendasi Menu:**
    - **Underweight:** Fokus pada energi dan protein tinggi
    - **Normal:** Gizi seimbang, protein cukup
    - **Overweight/Obesity:** Serat tinggi, energi rendah
    """.format((prioritas/len(df_bmi))*100, accuracy))

st.caption("Copyright © 2026 | MBG Smart Plate - Sistem Personalisasi Menu Gizi | v2.0")

# ============================================================
# SIDEBAR INFORMATION
# ============================================================
with st.sidebar.expander("ℹ️ Tentang Aplikasi"):
    st.markdown("""
    **MBG Smart Plate** adalah sistem personalisasi menu gizi berbasis web yang mendukung **Program Makan Bergizi Gratis (MBG)** dari pemerintah.
    
    **Fitur:**
    - Klasifikasi status gizi otomatis (Underweight, Normal, Overweight, Obesity)
    - Rekomendasi bahan pangan berdasarkan status gizi
    - Berbasis data TKPI (Tabel Komposisi Pangan Indonesia)
    - Akurasi model Random Forest > 95%
    
    **Sumber Data:**
    - Data Antropometri (BMI)
    - TKPI Kemenkes RI
    """)
