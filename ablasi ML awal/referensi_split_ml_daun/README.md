# Acuan pembagian data eksperimen daun

Acuan untuk `EDA_Ablasi_ML_Daun_Error_Analysis_Lengkap.ipynb`, berdasarkan notebook `EDA_Ablasi_ML_Daun.ipynb`.

- Seed 42, outer 5 fold dan inner 3 fold
- Seed inner 43, 44, 45, 46, 47
- `manifest.csv`: urutan sumber, nama foto, label, hash piksel dan grup
- `outer_splits.csv`: identik dengan split hasil baseline dan hasil augmentasi sebelumnya
- `inner_splits.csv`: direkonstruksi dari fungsi split notebook lama, urutan data lama, dan seed inner yang sama. Inner split tidak disimpan pada eksperimen awal
- `split_config.json`: asal acuan dan hash integritas berkas

Notebook memuat kedua tingkat split langsung dari berkas tersebut, tanpa mengacak ulang. Pemeriksaan foto, label dan grup dilakukan sebelum training. Snapshot dibuat tanpa menjalankan training. Jangan mengganti foto atau mengedit acuan ini untuk memaksakan perbandingan skor. Dataset baru memerlukan versi eksperimen dan acuan baru.
