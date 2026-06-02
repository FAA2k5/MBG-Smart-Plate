# MBG Smart Plate - Sistem Personalisasi Menu Gizi Berbasis Web

## Setup Environment - Shell/Terminal
```bash
pip install pandas numpy matplotlib seaborn streamlit scikit-learn scipy statsmodels openpyxl
```

## Run Streamlit App
streamlit run dashboard.py

## Tujuan Analisis
### Problem 1: Parameter Gizi yang Paling Berpengaruh untuk Rekomendasi Menu Personalisasi
Tujuan analisis ini adalah mengidentifikasi parameter gizi (Energi, Protein, Lemak, Karbohidrat, Serat) yang paling berpengaruh dan fleksibel untuk dijadikan patokan dalam menyusun rekomendasi menu personalisasi berdasarkan status gizi siswa. Teknik yang digunakan adalah Coefficient of Variation (CV) dan literatur gizi anak sekolah.

### Problem 2: Variabel Dominan Klasifikasi Status Gizi Siswa
Tujuan analisis ini adalah menentukan variabel antropometri (Usia, Jenis Kelamin, Tinggi Badan, Berat Badan) yang paling dominan mempengaruhi klasifikasi status gizi siswa. Teknik yang digunakan adalah Feature Importance dengan algoritma Random Forest Classifier.

### Problem 3: Distribusi Status Gizi dan Prioritas Intervensi
Tujuan analisis ini adalah mengetahui distribusi status gizi siswa (Underweight, Normal, Overweight, Obesity), menghitung persentase siswa yang memerlukan intervensi prioritas, serta mengidentifikasi kelompok usia mana yang paling membutuhkan perhatian. Teknik yang digunakan adalah Crosstab Analysis dan Quantile-based Age Grouping.

### Problem 4: Rekomendasi Bahan Pangan Berdasarkan Status Gizi
Tujuan analisis ini adalah merekomendasikan bahan pangan dari database TKPI (Tabel Komposisi Pangan Indonesia) yang sesuai untuk setiap kategori status gizi: menaikkan berat badan (Underweight), mempertahankan berat badan ideal (Normal), dan menurunkan berat badan (Overweight/Obesity). Teknik yang digunakan adalah Filtering & Sorting berdasarkan parameter energi, protein, dan serat.

### Problem 5: Uji Efektivitas Menu Personalisasi (A/B Testing)
Tujuan analisis ini adalah menguji apakah terdapat hubungan signifikan antara jenis menu yang diberikan (Menu Standar MBG vs Menu Personalisasi MBG Smart Plate) dengan perbaikan status gizi siswa setelah periode intervensi 3 bulan. Teknik yang digunakan adalah Chi-Square Test of Independence dengan ukuran efek Cramer's V.

## Fitur Dashboard
- Key Performance Indicators: Total siswa, jumlah kategori status gizi, akurasi model, total bahan pangan, siswa prioritas intervensi
- Distribusi Status Gizi Siswa: Pie chart dan bar chart distribusi 4 kategori status gizi
- Analisis per Kelompok Usia: Distribusi status gizi berdasarkan kelompok usia (Quantile)
- Prediksi Status Gizi Individu: Input data antropometri untuk klasifikasi otomatis
- Rekomendasi Bahan Pangan: Rekomendasi menu per kategori status gizi (Underweight, Normal, Overweight)
- Visualisasi Bahan Pangan: Top 10 bahan dengan energi tertinggi dan serat tertinggi
- Evaluasi Model Klasifikasi: Akurasi model, classification report, confusion matrix
- Feature Importance: Visualisasi faktor penentu status gizi (Berat Badan, Tinggi Badan, Usia, Jenis Kelamin)
- A/B Testing: Hasil uji Chi-Square efektivitas menu personalisasi

## Dataset yang Digunakan
- Dataset Antropometri (BMI)
- Dataset Menu


