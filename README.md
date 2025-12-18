# 🌴 Sistem Rekomendasi Destinasi Wisata Lampung Provinsi Lampung menggunakan **Content-Based Filtering (CBF)** dan **Location-Based Service (LBS)**.
## Team-2-PSD-RA

## 📋 Deskripsi Proyek
Sistem ini membantu wisatawan menemukan destinasi wisata di Lampung yang sesuai dengan preferensi pribadi dengan mempertimbangkan kedekatan lokasi geografis untuk perencanaan perjalanan yang efisien.

## ✨ Fitur Utama
- ✅ Rekomendasi berdasarkan kategori wisata (alam, buatan, budaya, religi, lainnya)
- ✅ Rekomendasi berdasarkan fasilitas (parkir, restoran, toilet, mushola)
- ✅ Rekomendasi berdasarkan lokasi untuk perencanaan perjalanan ke beberapa destinasi wisata
- ✅ Dashboard interaktif berbasis Streamlit

## 📊 Dataset
- **Sumber**: SISPARNAS & Google Maps
- **Jumlah data**: 439 destinasi wisata
- **Atribut**: nama destinasi, kategori, lokasi, fasilitas, rating, koordinat geografis

## 🎯 Sasaran Proyek
- Minimal 5 rekomendasi relevan perpengguna
- Dashboard interaktif yang user-friendly dengan response time ≤ 2 detik

## 📝 Metodologi
Proyek ini menggunakan **CRISP-DM** dengan tahapan:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

## 📁 Struktur Proyek
```
├── data/                    # Dataset wisata
├── app.py                   # Aplikasi Streamlit
├── model.py                 # preprocessing & Model 
├── requirements.txt         # Dependencies
└── README.md
```
## 📈 Hasil Evaluasi Sistem Rekomendasi
### Content-Based Filtering (CBF)
- Intra Mean: 1.0
- Inter Mean: 0.0
- separation: 1
- Coverage: 100%
- Diversity: 0

### Hybrid (CBF + LBS)
- Intra Mean: 0.782
- Inter Mean: 0.278
- Separation: 0.504
- Coverage: 100%
- Diversity: 0.173

## 👥 Tim Pengembang
- **Ukasyah Muntaha** (122450028) - Data Engineer
- **Syalaisha Andina Putriansyah** (122450121) - Data Analyst  
- **Nurul Alfajar Gumel** (122450127) - Data Scientist

---
**Proyek Sains Data - Fakultas Sains ITERA**
