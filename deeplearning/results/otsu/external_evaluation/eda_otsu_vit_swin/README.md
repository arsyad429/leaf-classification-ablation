# EDA Otsu untuk ViT-B/16 dan Swin-T pada evaluasi eksternal

Analisis ini memasangkan prediksi per foto dari kondisi **N** dan **S-O** untuk ViT-B/16 dan Swin-T pretrained. Keduanya dilatih pada 60 foto primer. Otsu diterapkan pada data latih, sedangkan kedua kondisi memakai 150 foto sekunder yang sama untuk evaluasi. Fokus utama memakai augmentasi OFF. Hasil augmentasi ON disertakan untuk melihat apakah polanya bertahan.

## Perubahan performa

| Model | Augmentasi | Benar N → S-O | Macro-F1 N → S-O | Jadi benar | Jadi salah |
|---|---|---:|---:|---:|---:|
| ViT-B/16 | OFF | 94 → 114 | 0,6053 → 0,7656 | 29 | 9 |
| Swin-T | OFF | 103 → 118 | 0,6844 → 0,7789 | 28 | 13 |
| ViT-B/16 | ON | 92 → 102 | 0,6110 → 0,6723 | 20 | 10 |
| Swin-T | ON | 101 → 102 | 0,6645 → 0,7047 | 33 | 32 |

### Perbaikan terbesar ada di kamboja dan aglaonema

Pada augmentasi OFF, jumlah foto **kamboja jepang** yang benar naik dari **6 ke 17 dari 25** pada ViT dan dari **7 ke 19** pada Swin. **Aglaonema** naik dari 10 ke 18 pada ViT dan dari 13 ke 19 pada Swin. Sebanyak **17 foto membaik pada kedua model**, termasuk 8 kamboja, 4 aglaonema, 3 peperomia, dan 2 nangka. [Grafik recall per kelas](01_recall_per_kelas.png) · [Pasangan per foto](perbandingan_per_foto.csv)

Kedua model juga lebih jarang keliru memberi label **paku tanduk rusa**. False positive paku turun dari **21 ke 6** pada ViT dan dari **33 ke 9** pada Swin. Untuk ViT, benar pada kelas paku sendiri ikut turun dari 22 ke 19. Untuk Swin, benar pada paku naik dari 24 ke 25, sementara benar pada lidah metua turun dari 19 ke 12. Saat augmentasi ON, Swin memberi label kamboja pada **66 foto** setelah Otsu, dengan 24 benar. Ini membantu menjelaskan mengapa tambahan foto benar bersih pada Swin tinggal satu. [Tabel per kelas](ringkasan_per_kelas.csv)

### Apa yang terlihat pada input latih

Mask S-O mempertahankan median **45,1% piksel** dari 60 foto primer. Median per kelas lebih kecil pada kamboja (**28,2%**) dan lidah metua (**31,9%**). Panel latih memperlihatkan bahwa mask berbasis warna dapat menekan bagian latar pada beberapa foto. Mask juga bisa menyertakan benda berwarna serupa atau membuang bagian daun. Luas mask mengukur area yang dipertahankan. Untuk mengukur ketepatan terhadap daun diperlukan mask anotasi. [Distribusi fraksi mask](02_fraksi_mask_otsu_primer.png) · [Panel mask pada foto latih](../../benchmark/figures/06_panel_otsu_training.png) · [CSV audit mask](../../benchmark/mask_audit.csv)

**Penjelasan yang konsisten dengan data** adalah Otsu saat training mengubah komposisi visual yang dipelajari model. Di set sekunder, model kemudian lebih jarang memilih paku untuk kelas lain dan lebih sering mengenali kamboja. Galeri memperlihatkan dua kamboja dengan daun terlihat di antara pot, batang, atau latar yang ramai. Satu contoh aglaonema yang membaik memperlihatkan pola belang daun yang jelas. Pada contoh nangka yang memburuk, kedua model justru berpindah dari nangka ke kamboja. Foto-foto ini menunjukkan perubahan prediksi yang teramati. Bukti bahwa model berpindah dari latar ke daun memerlukan pemeriksaan tambahan. [Galeri foto yang berubah](03_galeri_perubahan_prediksi.png) · [Atribusi foto](atribusi_foto.csv)

### Batas kesimpulan

Perbandingan ini eksploratif pada satu seed dan satu set uji. Pada augmentasi OFF, kedua model memakai learning rate `1e-4` pada N dan S-O, tetapi durasi berubah dari **20 ke 8 epoch** pada ViT dan **12 ke 8 epoch** pada Swin. Selisih yang tercatat mencakup PCD dan durasi hasil tuning. Kontrol dengan learning rate, epoch, dan seed yang sama diperlukan untuk mengukur kontribusi PCD secara lebih terisolasi.

Sumber prediksi dan recipe tersedia di [hasil eksternal awal](../../../external_evaluation/) dan [hasil eksternal Otsu](../). [Manifest foto sekunder](../../../../../Dataset_Sekunder/metadata.csv) mencatat sumber dan atribusi per foto.

## Regenerasi

Jalankan dari checkout repo dengan Python 3.10+, NumPy, Matplotlib, dan Pillow. Gunakan ZIP kurasi 150 foto sekunder. Skrip memverifikasi hash semua foto yang ditampilkan.

```bash
python -B deeplearning/results/otsu/external_evaluation/eda_otsu_vit_swin/generate_eda.py --dataset-zip /path/Dataset_Sekunder_Evaluasi_25_per_kelas.zip
```
