import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import chi2_contingency
import warnings
import os

warnings.filterwarnings('ignore')

# Set style
sns.set(style='whitegrid')
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_bmi_data():
    """Memuat dataset antropometri BMI (sama seperti di notebook)"""
    
    # URL dari notebook
    url_bmi = 'https://docs.google.com/spreadsheets/d/1E-sdZCDNeM-6gXMc72xx4si10InHdp5DKERisgkzDSs/export?format=csv'
    
    try:
        df = pd.read_csv(url_bmi)
        
        # Hitung BMI (sama seperti di notebook)
        df['BMI'] = round(df['Weight'] / ((df['Height'] / 100) ** 2), 2)
        
        # Buat kelompok usia berdasarkan kuartil (sama seperti di notebook)
        df['Age_Group'] = pd.qcut(df['Age'], q=4, labels=['Q1 (6-7 thn)', 'Q2 (8-9 thn)', 'Q3 (10-11 thn)', 'Q4 (12 thn)'])
        
        st.success(f"✅ Load data BMI: {df.shape[0]} siswa")
        return df
    except Exception as e:
        st.error(f"❌ Gagal memuat data BMI: {e}")
        st.stop()

@st.cache_data
def load_tkpi_data():
    """Memuat dataset TKPI (sama seperti di notebook)"""
    
    # URL dari notebook
    url_tkpi = 'https://docs.google.com/spreadsheets/d/1pTP3BZiwKkkdYRG1ky8c6P1_Iie7PUiVjman4XAtdBs/export?format=csv'
    
    try:
        df = pd.read_csv(url_tkpi)
        st.success(f"✅ Load data TKPI: {df.shape[0]} bahan")
        return df
    except Exception as e:
        st.error(f"❌ Gagal memuat data TKPI: {e}")
        st.stop()

# ============================================================
# FUNGSI REKOMENDASI BAHAN (BERDASARKAN ANALISIS NOTEBOOK)
# ============================================================
def get_rekomendasi_by_status(status, df_tkpi):
    """
    Memberikan rekomendasi bahan pangan berdasarkan status gizi
    Berdasarkan analisis Problem 4 di notebook
    """
    df_filtered = df_tkpi[df_tkpi['ENERGI (Kal)'] > 0].copy()
    
    if status == 'Underweight':
        # Prioritas: Energi Tinggi, Protein Tinggi
        rekomendasi = df_filtered.nlargest(10, 'ENERGI (Kal)')[
            ['NAMA BAHAN', 'KATEGORI', 'ENERGI (Kal)', 'PROTEIN (g)', 'LEMAK (g)', 'KH (g)']
        ]
        target = "Meningkatkan berat badan"
        strategi = "Tingkatkan asupan energi dan protein"
        warna = "#E74C3C"
        
    elif status in ['Overweight', 'Obesity']:
        # Prioritas: Serat Tinggi, Energi Rendah
        rekomendasi = df_filtered.nlargest(10, 'SERAT (g)')[
            ['NAMA BAHAN', 'KATEGORI', 'SERAT (g)', 'ENERGI (Kal)', 'PROTEIN (g)', 'KH (g)']
        ]
        target = "Mengontrol/Menurunkan berat badan"
        strategi = "Tingkatkan asupan serat, kurangi kalori dan lemak"
        warna = "#2ECC71"
        
    else:  # Normal
        # Prioritas: Gizi Seimbang, Protein cukup
        rekomendasi = df_filtered.nlargest(10, 'PROTEIN (g)')[
            ['NAMA BAHAN', 'KATEGORI', 'PROTEIN (g)', 'ENERGI (Kal)', 'SERAT (g)', 'LEMAK (g)']
        ]
        target = "Mempertahankan berat badan ideal"
        strategi = "Pola makan gizi seimbang (karbohidrat + lauk + sayur + buah)"
        warna = "#3498DB"
    
    return rekomendasi, target, strategi, warna

def get_rekomendasi_by_kategori(kategori, df_tkpi):
    """
    Memberikan rekomendasi bahan berdasarkan kategori (KARBOHIDRAT, LAUK, SAYURAN, BUAH, SUSU)
    """
    df_filtered = df_tkpi[df_tkpi['ENERGI (Kal)'] > 0].copy()
    
    if kategori in df_filtered['KATEGORI'].values:
        kategori_foods = df_filtered[df_filtered['KATEGORI'] == kategori]
        if kategori in ['LAUK', 'SUSU']:
            rekomendasi = kategori_foods.nlargest(5, 'PROTEIN (g)')[['NAMA BAHAN', 'PROTEIN (g)', 'ENERGI (Kal)']]
        else:
            rekomendasi = kategori_foods.nlargest(5, 'ENERGI (Kal)')[['NAMA BAHAN', 'ENERGI (Kal)', 'PROTEIN (g)']]
        return rekomendasi
    return pd.DataFrame()

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
    
    return model, scaler, le, accuracy, X_test_scaled, y_test, y_pred, model.feature_importances_

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
# FUNGSI CHI-SQUARE TEST (Problem 5)
# ============================================================
def run_chi_square_test():
    """
    Melakukan uji Chi-Square untuk Problem 5
    Menu Standar vs Menu Personalisasi
    """
    # Parameter eksperimen (sama seperti di notebook)
    jumlah_siswa_per_grup = 200
    tingkat_perbaikan_A = 0.30
    tingkat_perbaikan_B = 0.85
    
    # Simulasi data
    np.random.seed(42)
    perbaikan_A = np.random.binomial(n=1, p=tingkat_perbaikan_A, size=jumlah_siswa_per_grup).sum()
    perbaikan_B = np.random.binomial(n=1, p=tingkat_perbaikan_B, size=jumlah_siswa_per_grup).sum()
    
    tidak_membaik_A = jumlah_siswa_per_grup - perbaikan_A
    tidak_membaik_B = jumlah_siswa_per_grup - perbaikan_B
    
    tabel_kontingensi = [
        [perbaikan_A, tidak_membaik_A],
        [perbaikan_B, tidak_membaik_B]
    ]
    
    chi2, p_value, dof, expected = chi2_contingency(tabel_kontingensi)
    
    # Cramer's V
    n = sum(sum(row) for row in tabel_kontingensi)
    phi2 = chi2 / n
    cramers_v = np.sqrt(phi2 / min(2-1, 2-1))
    
    return {
        'tabel': tabel_kontingensi,
        'perbaikan_A': perbaikan_A,
        'perbaikan_B': perbaikan_B,
        'tingkat_A': perbaikan_A / jumlah_siswa_per_grup,
        'tingkat_B': perbaikan_B / jumlah_siswa_per_grup,
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'expected': expected,
        'cramers_v': cramers_v
    }

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
    model, scaler, le, accuracy, X_test, y_test, y_pred, feature_importances = train_model(df_bmi)

# Sidebar
with st.sidebar:
    st.image("https://www.bgn.go.id/bgn_logo/BGN_LOGO_BW_WHITE.png", width=80)
    st.title("MBG Smart Plate")
    st.markdown("### Sistem Personalisasi Menu Gizi")
    st.markdown("**Program Makan Bergizi Gratis (MBG)**")
    st.markdown("Badan Gizi Nasional")
    st.markdown("---")
    st.markdown("**📊 Informasi Data**")
    st.markdown(f"- **Total Siswa:** {len(df_bmi):,}")
    st.markdown(f"- **Total Bahan:** {len(df_tkpi):,}")
    st.markdown(f"- **Akurasi Model:** {accuracy:.2%}")
    st.markdown("---")
    st.caption("© 2026 MBG Smart Plate")

# Main content
st.title("🍽️ MBG Smart Plate")
st.markdown("### Dashboard Sistem Personalisasi Menu Gizi Berbasis Web")
st.markdown("*Mendukung Program Makan Bergizi Gratis (MBG) Badan Gizi Nasional*")

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
    pct_prioritas = (prioritas / len(df_bmi)) * 100
    st.metric("Siswa Prioritas Intervensi", f"{prioritas:,} ({pct_prioritas:.1f}%)")

st.markdown("---")

# ============================================================
# TAB 1: DISTRIBUSI STATUS GIZI (PROBLEM 3)
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Problem 1-3", "🍎 Problem 4", "📊 Problem 5", "🎯 Prediksi Status", "📉 Model Performance"
])

with tab1:
    st.subheader("Problem 1-3: Analisis Distribusi dan Faktor Dominan Status Gizi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart Status Gizi
        status_counts = df_bmi['Status_Gizi'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors_status = {
            'Underweight': '#E74C3C',
            'Normal': '#2ECC71',
            'Overweight': '#F39C12',
            'Obesity': '#C0392B'
        }
        pie_colors = [colors_status.get(s, '#3498DB') for s in status_counts.index]
        
        wedges, texts, autotexts = ax.pie(
            status_counts.values,
            labels=status_counts.index,
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=90,
            explode=[0.03] * len(status_counts)
        )
        ax.set_title('Distribusi Status Gizi Siswa', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        # Bar chart Status Gizi
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(status_counts.index, status_counts.values, 
                      color=[colors_status.get(s, '#3498DB') for s in status_counts.index],
                      edgecolor='black')
        ax.set_title('Jumlah Siswa per Status Gizi', fontsize=14, fontweight='bold')
        ax.set_xlabel('Status Gizi', fontsize=12)
        ax.set_ylabel('Jumlah Siswa', fontsize=12)
        
        for bar, count in zip(bars, status_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   str(count), ha='center', fontweight='bold')
        
        st.pyplot(fig)
        plt.close(fig)
    
    # Detail distribusi
    st.subheader("Detail Distribusi Status Gizi")
    status_df = df_bmi['Status_Gizi'].value_counts().reset_index()
    status_df.columns = ['Status Gizi', 'Jumlah Siswa']
    status_df['Persentase'] = (status_df['Jumlah Siswa'] / len(df_bmi) * 100).round(1).astype(str) + '%'
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    
    # Kelompok usia prioritas
    st.subheader("Kelompok Usia Prioritas Intervensi")
    
    age_priority = df_bmi[df_bmi['Status_Gizi'] != 'Normal'].groupby('Age_Group').size()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        age_colors = ['#E74C3C', '#F39C12', '#2ECC71', '#3498DB']
        bars = ax.bar(age_priority.index, age_priority.values, color=age_colors[:len(age_priority)], edgecolor='black')
        ax.set_title('Kelompok Usia dengan Masalah Gizi', fontsize=14, fontweight='bold')
        ax.set_xlabel('Kelompok Usia', fontsize=12)
        ax.set_ylabel('Jumlah Siswa Bermasalah', fontsize=12)
        
        for bar, count in zip(bars, age_priority.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   str(count), ha='center', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.info(f"""
        **📌 Kesimpulan Problem 3:**
        
        - **Siswa Prioritas Intervensi:** {prioritas:,} siswa ({pct_prioritas:.1f}%)
        - **Underweight:** {status_counts.get('Underweight', 0)} siswa
        - **Overweight:** {status_counts.get('Overweight', 0)} siswa
        - **Obesity:** {status_counts.get('Obesity', 0)} siswa
        
        **Kelompok usia prioritas:** {age_priority.idxmax() if len(age_priority) > 0 else 'Tidak ada'}
        """)
    
    # Feature Importance (Problem 2)
    st.subheader("Feature Importance - Faktor Dominan Status Gizi (Problem 2)")
    
    feature_names = ['Usia', 'Tinggi Badan', 'Berat Badan', 'Jenis Kelamin']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.argsort(feature_importances)[::-1]
    colors_fi = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    
    bars = ax.bar([feature_names[i] for i in indices], feature_importances[indices], color=colors_fi, edgecolor='black')
    ax.set_title('Feature Importance - Faktor Penentu Status Gizi', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fitur', fontsize=12)
    ax.set_ylabel('Nilai Importance', fontsize=12)
    
    for bar, imp in zip(bars, feature_importances[indices]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{imp:.3f} ({imp*100:.1f}%)', ha='center', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    st.info("""
    **📌 Kesimpulan Problem 2:**
    - **Berat Badan** adalah faktor paling dominan (64.2%)
    - **Tinggi Badan** berkontribusi 27.7%
    - **Usia** berkontribusi 7.8%
    - **Jenis Kelamin** berkontribusi 0.3%
    """)
    
    # Parameter Gizi Prioritas (Problem 1)
    st.subheader("Parameter Gizi Prioritas (Problem 1)")
    
    # Hitung Coefficient of Variation untuk parameter gizi
    numeric_cols = ['ENERGI (Kal)', 'PROTEIN (g)', 'LEMAK (g)', 'KH (g)', 'SERAT (g)']
    available_cols = [col for col in numeric_cols if col in df_tkpi.columns]
    
    if available_cols:
        stats_df = df_tkpi[available_cols].agg(['std', 'mean']).T
        stats_df['cv'] = stats_df['std'] / stats_df['mean']
        stats_df = stats_df.sort_values('cv', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(stats_df.index, stats_df['cv'].values, color='skyblue', edgecolor='black')
        ax.set_title('Coefficient of Variation - Parameter Gizi TKPI', fontsize=14, fontweight='bold')
        ax.set_xlabel('Coefficient of Variation (CV)', fontsize=12)
        ax.set_ylabel('Parameter Gizi', fontsize=12)
        
        for bar, cv in zip(bars, stats_df['cv'].values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{cv:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.info("""
        **📌 Kesimpulan Problem 1:**
        - **ENERGI (Kal)** - Penentu berat badan (prioritas utama)
        - **PROTEIN (g)** - Pertumbuhan dan rasa kenyang
        - **KH (g)** - Sumber energi utama
        - **LEMAK (g)** - Densitas energi tinggi
        - **SERAT (g)** - Kontrol berat badan dan pencernaan
        """)

with tab2:
    st.subheader("Problem 4: Rekomendasi Menu Berdasarkan Status Gizi")
    
    # Pilih kategori
    kategori_options = ['Underweight', 'Normal', 'Overweight', 'Obesity']
    selected_kategori = st.selectbox("Pilih Kategori Status Gizi", kategori_options)
    
    rekomendasi, target, strategi, warna = get_rekomendasi_by_status(selected_kategori, df_tkpi)
    
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
        sns.barplot(data=rekomendasi.head(8), y='NAMA BAHAN', x='ENERGI (Kal)', 
                   palette='Reds_r', ax=ax)
        ax.set_title(f'Rekomendasi untuk {selected_kategori}: Bahan dengan Energi Tertinggi', fontsize=12)
        ax.set_xlabel('Energi (Kalori)', fontsize=10)
        
    elif selected_kategori in ['Overweight', 'Obesity']:
        sns.barplot(data=rekomendasi.head(8), y='NAMA BAHAN', x='SERAT (g)', 
                   palette='Greens_r', ax=ax)
        ax.set_title(f'Rekomendasi untuk {selected_kategori}: Bahan dengan Serat Tertinggi', fontsize=12)
        ax.set_xlabel('Serat (gram)', fontsize=10)
        
    else:
        sns.barplot(data=rekomendasi.head(8), y='NAMA BAHAN', x='PROTEIN (g)', 
                   palette='Blues_r', ax=ax)
        ax.set_title(f'Rekomendasi untuk {selected_kategori}: Bahan dengan Protein Tertinggi', fontsize=12)
        ax.set_xlabel('Protein (gram)', fontsize=10)
    
    ax.set_ylabel('Nama Bahan', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Rekomendasi per kategori
    st.subheader("Rekomendasi Berdasarkan Kategori Bahan")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🍚 KARBOHIDRAT**")
        rek_karb = get_rekomendasi_by_kategori('KARBOHIDRAT', df_tkpi)
        if not rek_karb.empty:
            st.dataframe(rek_karb, hide_index=True, use_container_width=True)
        else:
            st.caption("Tidak ada data")
    
    with col2:
        st.markdown("**🍗 LAUK**")
        rek_lauk = get_rekomendasi_by_kategori('LAUK', df_tkpi)
        if not rek_lauk.empty:
            st.dataframe(rek_lauk, hide_index=True, use_container_width=True)
        else:
            st.caption("Tidak ada data")
    
    with col3:
        st.markdown("**🥬 SAYURAN**")
        rek_sayur = get_rekomendasi_by_kategori('SAYURAN', df_tkpi)
        if not rek_sayur.empty:
            st.dataframe(rek_sayur, hide_index=True, use_container_width=True)
        else:
            st.caption("Tidak ada data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🍎 BUAH**")
        rek_buah = get_rekomendasi_by_kategori('BUAH', df_tkpi)
        if not rek_buah.empty:
            st.dataframe(rek_buah, hide_index=True, use_container_width=True)
        else:
            st.caption("Tidak ada data")
    
    with col2:
        st.markdown("**🥛 SUSU**")
        rek_susu = get_rekomendasi_by_kategori('SUSU', df_tkpi)
        if not rek_susu.empty:
            st.dataframe(rek_susu, hide_index=True, use_container_width=True)
        else:
            st.caption("Tidak ada data")

with tab3:
    st.subheader("Problem 5: Uji Efektivitas Menu Personalisasi (A/B Testing)")
    
    # Run Chi-Square test
    hasil = run_chi_square_test()
    
    st.markdown("""
    **Desain Eksperimen:**
    - **Grup A (Kontrol):** Menu Standar MBG
    - **Grup B (Perlakuan):** Menu Personalisasi MBG Smart Plate
    - **Jumlah siswa per grup:** 200 siswa
    - **Durasi:** 3 bulan
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #E74C3C20; padding: 15px; border-radius: 10px;">
            <h4 style="color: #E74C3C;"> Grup A (Menu Standar)</h4>
            <p>Jumlah siswa: 200</p>
            <p>Siswa yang membaik: <strong>{hasil['perbaikan_A']} siswa</strong></p>
            <p>Siswa yang tidak membaik: <strong>{200 - hasil['perbaikan_A']} siswa</strong></p>
            <p>Tingkat perbaikan: <strong>{hasil['tingkat_A']*100:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: #2ECC7120; padding: 15px; border-radius: 10px;">
            <h4 style="color: #2ECC71;"> Grup B (Menu Personalisasi)</h4>
            <p>Jumlah siswa: 200</p>
            <p>Siswa yang membaik: <strong>{hasil['perbaikan_B']} siswa</strong></p>
            <p>Siswa yang tidak membaik: <strong>{200 - hasil['perbaikan_B']} siswa</strong></p>
            <p>Tingkat perbaikan: <strong>{hasil['tingkat_B']*100:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabel kontingensi
    st.subheader("Tabel Kontingensi (2x2)")
    tabel_df = pd.DataFrame(
        hasil['tabel'],
        index=['Menu Standar (A)', 'Menu Personalisasi (B)'],
        columns=['Membaik', 'Tidak Membaik']
    )
    st.dataframe(tabel_df, use_container_width=True)
    
    # Hasil Uji Chi-Square
    st.subheader("Hasil Uji Chi-Square")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Chi-Square Statistic", f"{hasil['chi2']:.4f}")
        st.metric("Degrees of Freedom", hasil['dof'])
    
    with col2:
        st.metric("P-value", f"{hasil['p_value']:.6f}")
        st.metric("Cramer's V", f"{hasil['cramers_v']:.4f}")
    
    # Interpretasi
    if hasil['p_value'] < 0.05:
        st.success(f"""
        ✅ **KEPUTUSAN: TOLAK H0**
        
        P-value ({hasil['p_value']:.6f}) < α (0.05)
        
        **Kesimpulan:** Terdapat hubungan yang SIGNIFIKAN antara jenis menu yang diberikan dengan perbaikan status gizi siswa.
        
        Menu Personalisasi (B) {hasil['tingkat_B']*100:.1f}% lebih efektif dibandingkan Menu Standar (A) yang hanya {hasil['tingkat_A']*100:.1f}%.
        
        **Rekomendasi:** Program MBG Smart Plate dapat dijalankan.
        """)
    else:
        st.warning(f"""
        ❌ **KEPUTUSAN: GAGAL TOLAK H0**
        
        P-value ({hasil['p_value']:.6f}) > α (0.05)
        
        Tidak ada bukti cukup bahwa menu personalisasi lebih efektif.
        """)
    
    # Expected frequencies
    expected_df = pd.DataFrame(
        hasil['expected'],
        index=['Menu Standar (A)', 'Menu Personalisasi (B)'],
        columns=['Membaik (Expected)', 'Tidak Membaik (Expected)']
    )
    st.subheader("Expected Frequencies (jika H0 benar)")
    st.dataframe(expected_df.round(2), use_container_width=True)
    
    # Visualisasi perbandingan
    st.subheader("Visualisasi Hasil Eksperimen")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart perbandingan
    labels = ['Menu Standar (A)', 'Menu Personalisasi (B)']
    rates = [hasil['tingkat_A'], hasil['tingkat_B']]
    colors_bar = ['#E74C3C', '#2ECC71']
    
    bars = axes[0].bar(labels, rates, color=colors_bar, edgecolor='black', alpha=0.8)
    axes[0].set_title('Perbandingan Tingkat Perbaikan Status Gizi', fontsize=12)
    axes[0].set_ylabel('Tingkat Perbaikan (%)', fontsize=10)
    axes[0].set_ylim(0, 1.0)
    
    for bar, rate in zip(bars, rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{rate*100:.1f}%', ha='center', fontweight='bold', fontsize=11)
    
    # Stacked bar chart
    membaik = [hasil['perbaikan_A'], hasil['perbaikan_B']]
    tidak_membaik = [200 - hasil['perbaikan_A'], 200 - hasil['perbaikan_B']]
    
    axes[1].bar(labels, membaik, label='Membaik', color='#2ECC71', edgecolor='black')
    axes[1].bar(labels, tidak_membaik, bottom=membaik, label='Tidak Membaik', color='#E74C3C', edgecolor='black')
    axes[1].set_title('Distribusi Hasil Eksperimen per Grup', fontsize=12)
    axes[1].set_ylabel('Jumlah Siswa', fontsize=10)
    axes[1].legend()
    
    for i, (m, tm) in enumerate(zip(membaik, tidak_membaik)):
        axes[1].text(i, m + tm + 2, f'n={m+tm}', ha='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab4:
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
            
            st.markdown("---")
            st.subheader("📋 Hasil Prediksi")
            
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
            
            rekomendasi, target, strategi, warna_rek = get_rekomendasi_by_status(status, df_tkpi)
            
            st.markdown(f"""
            <div style="background-color: {warna_rek}20; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                <p><strong>🎯 Target:</strong> {target}</p>
                <p><strong>📌 Strategi:</strong> {strategi}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(rekomendasi, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("📊 Performa Model Klasifikasi Status Gizi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Akurasi Model", f"{accuracy:.2%}")
        
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

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("© 2026 MBG Smart Plate - Sistem Personalisasi Menu Gizi Berbasis Web | Badan Gizi Nasional")