# Ringkasan error analysis klasifikasi daun
- Total gambar: 60; kelas: 6; outer folds: 5.
- Eksperimen utama: F4 (HSV + SIFT–BoVW), SVM; kondisi N/E/S-O/S-K/E-S-O/E-S-K.
- Benchmark utama menggunakan nested CV sesuai bagian 1–8; diagnosis terkontrol pada bagian 14–15 memakai tanpa augmentasi.
- Mask manual tersedia: 0/60. Mode background: auto_proxy.

## Transisi prediksi dibanding N (OOF baseline dari eksperimen asli)
- N: akurasi=0.967, loss=0, gain=0, p McNemar eksak eksploratif=1.0000.
- E: akurasi=0.917, loss=3, gain=0, p McNemar eksak eksploratif=0.2500.
- S-O: akurasi=0.750, loss=13, gain=0, p McNemar eksak eksploratif=0.0002.
- S-K: akurasi=0.850, loss=7, gain=0, p McNemar eksak eksploratif=0.0156.
- E-S-O: akurasi=0.783, loss=11, gain=0, p McNemar eksak eksploratif=0.0010.
- E-S-K: akurasi=0.833, loss=8, gain=0, p McNemar eksak eksploratif=0.0078.

## Jumlah SIFT dan separabilitas (deskriptif / train-reference)
- N: mean SIFT=299.5; silhouette train=0.098; Fisher train=0.535.
- E: mean SIFT=300.0; silhouette train=0.083; Fisher train=0.492.
- S-O: mean SIFT=95.1; silhouette train=-0.050; Fisher train=0.264.
- S-K: mean SIFT=160.3; silhouette train=-0.092; Fisher train=0.259.
- E-S-O: mean SIFT=111.5; silhouette train=-0.078; Fisher train=0.205.
- E-S-K: mean SIFT=183.7; silhouette train=-0.072; Fisher train=0.334.

## Interpretasi yang harus diuji
- Apakah mask menghapus bagian daun yang relevan? Verifikasi pada gambar dan anotasi ground truth.
- Apakah kehilangan SIFT serta penurunan separabilitas muncul pada foto yang berubah benar→salah?
- Apakah background-only tetap tinggi? Jika mask otomatis digunakan, hasil hanya proxy, belum bukti shortcut.
- Apakah shared vocabulary dan parameter tetap mengubah besaran kesenjangan N-vs-segmentasi?

## Batasan
- Audit fitur penuh dan scatter PCA 2D bersifat diagnosis, bukan pembuktian kausal.
- Seluruh uji statistik eksploratif; tidak ada koreksi multiple testing / validasi eksternal.
- Tanpa identitas individu tanaman untuk split berbasis group, hasil foto-level bisa optimistis.
- Evaluasi subset foto sedikit dan tidak seimbang; perbedaan 1–2 foto sangat memengaruhi skor.
- Jangan bandingkan indeks BoVW dari vocabulary berbeda secara langsung.
- Folder laporan: e:\PCD\Project Akhir\Notebook\reproduksi_github_f2473327_lengkap\error_analysis_tambahan
