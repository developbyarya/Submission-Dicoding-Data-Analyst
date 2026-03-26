# Panduan Menjalankan Streamlit App

Repo ini menyediakan aplikasi Streamlit untuk menampilkan visualisasi analisis e-commerce (Revenue, Top Produk, Segmentasi RFM/Sultan, dan Geospatial Potensi Customer).

## Prasyarat

- Python 3.9+ (disarankan 3.10 atau 3.11)
- Pastikan folder data `dataset/` sudah tersedia di root project

## Isi Folder Dataset yang Dibutuhkan

Aplikasi mengharapkan file CSV berikut di dalam folder `dataset/`:

- `orders_dataset.csv`
- `order_items_dataset.csv`
- `customers_dataset.csv`
- `geolocation_dataset.csv`
- `products_dataset.csv`
- `product_category_name_translation.csv`

## Instalasi Dependensi

1. Masuk ke folder project:

```bash
cd "Submission Dicoding Data Analyst"
```

2. (Opsional) Buat virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependensi:

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi Streamlit

Jalankan perintah berikut dari root project:

```bash
streamlit run dasboard/streamlit_app.py
```

Setelah itu, buka URL yang ditampilkan di terminal (biasanya `http://localhost:8501`).

## Catatan

- Komputasi awal mungkin membutuhkan waktu karena aplikasi memproses dan menyiapkan data.
- Hasil perhitungan tertentu di-cache menggunakan `st.cache_data` supaya akses berikutnya lebih cepat.

