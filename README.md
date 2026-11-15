# ML Studio

Aplikasi Streamlit untuk menganalisis dataset tabular (CSV/Excel): audit kualitas data, EDA, perbandingan
model machine learning yang jujur (baseline + cross-validation), catatan analis otomatis, laporan, dan skrip reproduksi.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tanpa data? Pilih salah satu dataset contoh (Wine untuk klasifikasi, Diabetes untuk regresi) di halaman awal.

## Alur kerja di aplikasi

| Tab | Isi |
|---|---|
| Ringkasan | KPI, skor kualitas data, temuan + saran, profil kolom |
| Eksplorasi | Statistik, distribusi (dengan catatan skew/outlier), korelasi, scatter, keterkaitan fitur vs target |
| Modeling | Persiapan otomatis, latih beberapa algoritma, leaderboard, evaluasi, feature importance, skrip, unduhan |
| Prediksi | Pakai model terbaik pada data baru (template CSV disediakan) |
| Panduan | Cara kerja, glosarium metrik, daftar algoritma, batasan |

## Keputusan metodologis (agar hasil bisa dipercaya)

- **Baseline naif** (kelas terbanyak / rata-rata) selalu ikut dilatih; model terbaik harus mengalahkannya dan baseline tidak pernah dipilih.
- **Peringkat memakai cross-validation** pada data latih. Data uji hanya untuk evaluasi akhir.
- **Deteksi overfitting**: kolom `Gap` = skor latih − skor uji; catatan analis memberi peringatan bila > 0.15.
- **Pemeriksaan kebocoran data**: fitur numerik yang hampir identik dengan target diberi peringatan.
- **Kolom konstan, ID, dan teks hampir-unik** dikeluarkan otomatis dan dilaporkan, bukan dibuang diam-diam.
- Model `.joblib` bisa dilatih ulang di seluruh data (opsi sidebar); angka evaluasi tetap dari data uji.
- Tidak ada tuning hiperparameter; semua model memakai pengaturan default yang wajar.

## Struktur proyek

```
app.py               orkestrasi UI
mlstudio/core.py     logika data & ML (tanpa Streamlit/Plotly), semua keputusan dicatat sebagai Finding
mlstudio/charts.py   pembuat grafik Plotly (data masuk, Figure keluar)
mlstudio/ui.py       CSS dan komponen tampilan
tests/test_core.py   22 tes pytest untuk core
.streamlit/config.toml
```

Alur data di `core.py`:

```
DataFrame -> profile_dataset / quality_findings
          -> prepare_data -> PreparedData -> split_data -> run_experiment -> ExperimentResult
                                                              |-> analyst_notes / build_report
                                                              |-> reproducible_script / bundle_to_bytes
                                                              `-> predict_new
```

## Menambah algoritma

Tambahkan satu `ModelSpec` di `model_zoo()` dalam `mlstudio/core.py`. Leaderboard, grafik, penjelasan di tab Panduan,
dan skrip reproduksi mengikuti otomatis. Jika kelasnya bukan dari `sklearn.ensemble/linear_model/tree/neighbors/svm/dummy`,
tambahkan modulnya di `_public_import()`.

## Pengujian

```bash
pip install pytest
python -m pytest -q
```

Tes mencakup profil/kualitas data, skrining fitur, persiapan data (termasuk kasus error), eksperimen
klasifikasi & regresi, missing value, laporan, prediksi, serta eksekusi nyata skrip reproduksi dan pembuktian
bahwa skor skrip sama dengan skor di leaderboard.
