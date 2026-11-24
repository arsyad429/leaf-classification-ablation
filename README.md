# Klasifikasi Tanaman dari Citra Daun

**EDA, ablasi pengolahan citra, serta klasifikasi dengan machine learning klasik dan deep learning.**

Proyek ini membandingkan SVM, Random Forest, ResNet-50, ViT-B/16, dan Swin-T untuk enam kelas tanaman. Dataset primer berisi **60 foto UB**. Dataset sekunder berisi **150 foto iNaturalist**, masing-masing 25 foto per kelas, untuk evaluasi eksternal.

| Evaluasi | Konfigurasi terbaik yang tercatat | Benar | Macro-F1 |
|---|---|---:|---:|
| ML klasik, OOF primer | `N–F4–SVM` dengan augmentasi, seri dengan `F7` dan `F8` | 58/60 | 0,9444 |
| Deep learning, OOF primer | Swin-T pretrained, `N`, augmentasi OFF | 60/60 | 1,0000 |
| Deep learning, eksternal | Swin-T pretrained, `S-O`, augmentasi OFF | 118/150 | 0,7789 |

Macro-F1 primer adalah rata-rata lima outer fold. Macro-F1 eksternal dihitung dari 150 prediksi model yang dilatih pada seluruh 60 foto primer.

[Notebook ML](ablasi%20ML%20awal/EDA_Ablasi_ML_Daun.ipynb) · [Hasil ML baseline](ablasi%20ML%20awal/hasil_ml_full/) · [Hasil ML augmentasi](ablasi%20ML%20awal/hasil_ml_full_aug/) · [Eksperimen deep learning](#deep-learning) · [Manifest sekunder](Dataset_Sekunder/metadata.csv)

## Tujuan eksperimen ML klasik

- Mengamati distribusi kelas, karakteristik citra, dan duplikat piksel melalui EDA.
- Membandingkan enhancement CLAHE, segmentasi Otsu, dan segmentasi K-Means.
- Mengukur kontribusi warna, Sobel, Canny, dan SIFT–Bag of Visual Words (BoVW).
- Membandingkan Support Vector Machine (SVM) dan Random Forest (RF) dengan nested cross-validation.
- Menelaah dampak augmentasi train-only, kesalahan klasifikasi, dan kontribusi blok fitur.

## Data

Audit yang tersimpan mencakup 60 citra berukuran 960 × 1280 piksel, tanpa duplikat piksel identik.

| Label pada metadata | Jumlah citra |
|---|---:|
| Aglaonema | 10 |
| Daun nangka | 9 |
| Kamboja jepang | 9 |
| Lidah metua | 12 |
| Paku tanduk rusa | 14 |
| Peperomium semangka | 6 |
| **Total** | **60** |

Nama kelas dipertahankan sesuai metadata eksperimen. **Dataset mentah tidak disertakan.** Gambar contoh tersedia dalam output notebook dan artefak visual.

Dataset sekunder mempunyai 150 observasi iNaturalist yang berbeda. Manifest mencatat label lokal, sumber, takson, status identifikasi, lisensi, atribusi, dan hash gambar. Delapan foto berlisensi CC BY-NC. Ikuti lisensi dan atribusi masing-masing foto. [Manifest sekunder](Dataset_Sekunder/metadata.csv) · [Catatan kurasi](Dataset_Sekunder/README.txt)

## Rancangan ablasi

### Kondisi citra

| Kode | Perlakuan |
|---|---|
| `N` | Tanpa enhancement atau segmentasi |
| `E` | Enhancement CLAHE pada kanal L* di ruang Lab |
| `S-O` | Segmentasi Otsu pada kanal a* Lab |
| `S-K` | Segmentasi K-Means di ruang Lab |
| `E-S-O` | CLAHE dilanjutkan segmentasi Otsu |
| `E-S-K` | CLAHE dilanjutkan segmentasi K-Means |

Parameter awal PCD: CLAHE clip limit 1,5 dengan tile 8 × 8, K-Means dengan 4 cluster, serta operasi morfologi opening/closing dengan kernel elips 3 × 3. Parameter ini dibekukan sebagai kandidat awal, bukan diklaim sebagai konfigurasi PCD terbaik. Sisi terpanjang citra pada full mode dibatasi 640 piksel.

### Paket fitur

| Paket | Fitur |
|---|---|
| `F0` | Warna |
| `F1` | Warna + Sobel |
| `F2` | Warna + Canny |
| `F3` | SIFT–BoVW |
| `F4` | Warna + SIFT–BoVW |
| `F5` | Canny + SIFT–BoVW |
| `F6` | Sobel + SIFT–BoVW |
| `F7` | Warna + Canny + SIFT–BoVW |
| `F8` | Warna + Sobel + SIFT–BoVW |

Warna direpresentasikan oleh histogram HSV dan statistik S/V. Sobel dan Canny menghasilkan statistik serta histogram arah gradien. Deskriptor SIFT dikodekan menjadi histogram BoVW; kamusnya dipelajari hanya dari citra asli pada subset train. Profil ekstraksi menentukan jumlah bin warna, ukuran kamus BoVW, dan threshold Canny sehingga dimensi fitur dapat berbeda antarfold.

### Model dan evaluasi

- **Nested CV:** 5 outer fold untuk evaluasi dan 3 inner fold untuk pemilihan parameter, dengan seed 42. Pada data ini, setiap outer fold berisi 48 foto train dan 12 foto evaluasi.
- **Model:** SVM RBF dan Random Forest dengan 200 pohon; masing-masing memakai 9 kandidat hyperparameter pada full mode.
- **Profil fitur:** dipilih melalui inner CV dengan anchor `N–F8–SVM`, lalu digunakan bersama oleh seluruh 108 kombinasi dalam outer fold yang sama.
- **Kontrol kebocoran:** imputer, scaler, model, dan kamus BoVW di-fit dari train. Validation dan outer-evaluation tetap berupa foto asli. Dukungan split berbasis `plant_id` tersedia jika metadata individu tanaman lengkap.
- **Artefak evaluasi:** setiap skenario menghasilkan 540 baris metrik fold dan 6.480 prediksi OOF, yaitu satu prediksi per foto untuk setiap kombinasi.

## Augmentasi

Skenario augmentasi menambahkan satu varian untuk masing-masing dari enam operasi: Gaussian noise, flip horizontal/vertikal, rotasi 90°/180°/270°, salt-and-pepper corruption, brightness, dan contrast. Dengan citra asli, satu foto train menghasilkan 7 sampel; 48 foto outer-train menjadi 336 sampel untuk fitting.

Augmentasi dilakukan **hanya pada train**, termasuk inner-train saat tuning. Gaussian noise memakai sigma 3–10, corruption fraction 0,002–0,008, brightness ±10–20, dan contrast factor 0,85–1,15. Konfigurasi lengkap tersedia di [augmentation_config.json](ablasi%20ML%20awal/hasil_ml_full_aug/augmentation_config.json).

![Contoh augmentasi untuk ilustrasi](ablasi%20ML%20awal/hasil_ml_full_aug/eda_augmentasi_illustration.png)

Panel di atas memakai mode **illustration** yang memperkuat efek untuk keperluan visualisasi. Intensitas pada panel ini tidak boleh dianggap sebagai parameter training; acuannya adalah konfigurasi JSON dan kode `augment_rgb`.

## Hasil eksperimen

Empat konfigurasi teratas dari masing-masing `summary_108.csv`:

| Skenario | Input | Paket | Model | Macro-F1, mean ± SD fold | Akurasi rata-rata fold |
|---|---|---|---|---:|---:|
| Baseline | `E` | `F4` | SVM | 0,9311 ± 0,1203 | 95,00% |
| Baseline | `N` | `F4` | SVM | 0,9311 ± 0,1203 | 95,00% |
| Baseline | `N` | `F7` | SVM | 0,9311 ± 0,1203 | 95,00% |
| Baseline | `N` | `F8` | SVM | 0,9311 ± 0,1203 | 95,00% |
| Augmentasi | `N` | `F4` | SVM | 0,9444 ± 0,1242 | 96,67% |
| Augmentasi | `N` | `F7` | SVM | 0,9444 ± 0,1242 | 96,67% |
| Augmentasi | `N` | `F8` | SVM | 0,9444 ± 0,1242 | 96,67% |
| Augmentasi | `N` | `F5` | SVM | 0,9300 ± 0,0741 | 93,33% |

Sumber: [ranking baseline](ablasi%20ML%20awal/hasil_ml_full/summary_108.csv) dan [ranking augmentasi](ablasi%20ML%20awal/hasil_ml_full_aug/summary_108.csv). Skor seri tidak menunjukkan satu konfigurasi lebih unggul dari konfigurasi lain.

Untuk `N–F4–SVM`, baseline menghasilkan 57/60 prediksi benar dan augmentasi 58/60. Macro-F1 yang dihitung dari **seluruh prediksi OOF yang digabung** masing-masing adalah **0,9473** dan **0,9622**. Nilai ini berbeda dari rata-rata macro-F1 antarfold pada tabel karena macro-F1 bukan metrik yang linear. SD pada tabel adalah variasi antarfold, bukan confidence interval.

Temuan deskriptif dari hasil tersebut:

- Warna + SIFT–BoVW (`F4`) menyamai skor paket `F7` dan `F8` yang menambahkan fitur edge pada konfigurasi teratas.
- Konfigurasi tanpa enhancement/segmentasi (`N`) termasuk yang terbaik; hasil ini belum menunjukkan kebutuhan segmentasi untuk mencapai skor tertinggi pada data yang tersedia.
- Skor terbaik skenario augmentasi lebih tinggi daripada baseline. Profil fitur dipilih ulang dalam tiap skenario, sehingga perbedaan tersebut mencerminkan keseluruhan prosedur, bukan isolasi efek augmentasi dengan profil tetap.

### Analisis kesalahan

Pada `N–F4–SVM` dengan augmentasi, dua foto yang salah diklasifikasikan adalah:

| Foto | Label sebenarnya | Prediksi |
|---|---|---|
| `kamboja_jepang_2.jpg` | Kamboja jepang | Lidah metua |
| `peperomium_semangka_3.jpg` | Peperomium semangka | Lidah metua |

![Confusion matrix OOF N–F4–SVM dengan augmentasi](ablasi%20ML%20awal/hasil_ml_full_aug/error_analysis_top4/01_N_F4_SVM/confusion_matrix.png)

Confusion matrix menampilkan hitungan absolut dan normalisasi per kelas. Laporan per kelas, contoh foto salah, dan perbandingan prediksi tersedia di [analisis empat konfigurasi teratas](ablasi%20ML%20awal/hasil_ml_full_aug/error_analysis_top4/).

### Feature importance

Permutation importance mengukur penurunan macro-F1 ketika fitur atau blok fitur diacak pada outer-evaluation. Analisis tambahan disediakan untuk `N–F7–SVM` dan `N–F8–SVM`, yang menyamai skor tertinggi selain `F4`.

![Permutation importance per fold N–F7–SVM dengan augmentasi](ablasi%20ML%20awal/hasil_ml_full_aug/feature_importance_best_non_f4/N_F7_SVM_importance_by_fold.png)

Lihat [ringkasan importance per blok](ablasi%20ML%20awal/hasil_ml_full_aug/feature_importance_best_non_f4/block_importance_summary.csv). Importance bersifat deskriptif, bukan kausal; fitur berkorelasi dapat saling menggantikan. Indeks word BoVW tidak dibandingkan langsung antarfold karena kamusnya berbeda.

## Deep learning

Benchmark memakai ResNet-50, ViT-B/16, dan Swin-T dengan bobot awal ImageNet-1K atau acak. Setiap model diuji dengan augmentasi ON dan OFF. Run awal meliputi `N`, `E`, dan `S-K` (**36 konfigurasi**). Ekstensi Otsu menambah `S-O` dan `E-S-O` (**24 konfigurasi**). Totalnya **60 konfigurasi per protokol evaluasi**.

CLAHE memakai kanal `L*` Lab, clip limit 1,5, dan tile 8 × 8. Otsu memilih ambang per foto pada kanal `a*` Lab dengan threshold biner terbalik. Mask segmentasi menjalani opening dan closing dengan kernel elips 3 × 3. Pada input DL, RGB di luar mask bernilai nol. Model menerima input 224 × 224 dengan resize dan padding yang mempertahankan proporsi. [Kode PCD awal](deeplearning/data.py) · [Kode Otsu](deeplearning/run_otsu.py)

| Protokol | Data latih dan evaluasi | Perlakuan citra |
|---|---|---|
| **Primer OOF** | Split ML yang sama. Lima outer fold dengan 48 foto train dan 12 evaluasi per fold. Tiga inner fold untuk memilih learning rate dan durasi. | Kondisi PCD berlaku pada train, validasi, dan evaluasi primer. Augmentasi hanya pada train. |
| **Evaluasi eksternal** | Fit ulang pada seluruh 60 foto primer. Evaluasi pada 150 foto sekunder yang sama, dengan 25 foto per kelas. | CLAHE, Otsu, K-Means, dan augmentasi hanya pada train. Semua foto sekunder memakai decode RGB, resize, padding, dan normalisasi input yang sama. |

Pada evaluasi eksternal, learning rate dipilih dari rata-rata macro-F1 inner fold primer. Jumlah epoch memakai median epoch terbaik untuk kandidat learning rate terpilih. Recipe dikunci sebelum evaluasi sekunder. Run memakai seed 42, AdamW, batch size 8, dan A100. Batas tuning adalah 20 epoch dengan patience 4 untuk pretrained dan 40 epoch dengan patience 6 untuk bobot acak. [Split primer](deeplearning/splits/) · [Konfigurasi benchmark awal](deeplearning/results/benchmark/run_config.json) · [Recipe eksternal Otsu](deeplearning/results/otsu/external_evaluation/run_config.json)

### Hasil dan bukti visual

| Run | Konfigurasi terbaik menurut macro-F1 | Benar | Macro-F1 | Artefak |
|---|---|---:|---:|---|
| Primer awal | Swin-T pretrained, `N`, Aug OFF | 60/60 | 1,0000 | [Ringkasan](deeplearning/results/benchmark/summary.csv) · [Gambar](deeplearning/results/benchmark/figures/) |
| Primer Otsu | ViT-B/16 pretrained, `E-S-O`, Aug OFF | 58/60 | 0,9708 | [Ringkasan](deeplearning/results/otsu/benchmark/summary.csv) · [Gambar](deeplearning/results/otsu/benchmark/figures/) |
| Eksternal awal | Swin-T pretrained, `N`, Aug OFF | 103/150 | 0,6844 | [Ringkasan](deeplearning/results/external_evaluation/summary.csv) · [Gambar](deeplearning/results/external_evaluation/figures/) |
| Eksternal Otsu | Swin-T pretrained, `S-O`, Aug OFF | 118/150 | 0,7789 | [Ringkasan](deeplearning/results/otsu/external_evaluation/summary.csv) · [Gambar](deeplearning/results/otsu/external_evaluation/figures/) |

Pada Swin-T pretrained tanpa augmentasi, `S-O` saat training memberi **15 prediksi benar tambahan** dibanding `N` pada foto sekunder yang sama. Di seluruh 12 pasangan dengan arsitektur, bobot awal, dan status augmentasi setara, `S-O` mengungguli `N` pada **5 pasangan** dengan rata-rata Δ macro-F1 **−0,0038**. Pada run awal, rata-rata Δ macro-F1 terhadap `N` adalah **−0,0027** untuk CLAHE dan **−0,0650** untuk K-Means. Perbandingan ini mencakup recipe yang dituning per kondisi dan satu seed. [Perbandingan eksternal awal](deeplearning/results/external_evaluation/pcd_deltas.csv) · [Perbandingan eksternal Otsu](deeplearning/results/otsu/external_evaluation/pcd_deltas.csv)

Penurunan Swin-T `N` dari 60/60 pada OOF primer menjadi 103/150 pada evaluasi eksternal bersifat deskriptif. Jumlah foto latih dan sumber foto uji sama-sama berubah. Evaluasi eksternal mengukur performa pada set iNaturalist terkurasi. Foto dan pengamat yang berbeda masih bisa memotret individu tanaman yang sama.

### Menjalankan pipeline DL

Jalankan dari root repo dengan CUDA, PyTorch, torchvision, NumPy, Pillow, OpenCV, scikit-learn, threadpoolctl, dan Matplotlib. Environment tercatat memakai PyTorch `1.13.1+cu116`, torchvision `0.14.1+cu116`, scikit-learn `1.2.1`, dan OpenCV `4.10.0`. Sediakan foto primer di `Dataset/<label>/<image>`. Untuk evaluasi eksternal, sediakan foto berdasarkan `download_url` pada manifest di `Dataset_Sekunder/<label>/<image>`.

| Langkah | Perintah |
|---|---|
| Verifikasi foto primer dan split | `python -B deeplearning/prepare_splits.py --check` |
| Smoke test benchmark awal | `python -B deeplearning/run_benchmark.py --smoke` |
| Benchmark primer awal | `python -B deeplearning/run_benchmark.py` |
| Evaluasi eksternal awal | `python -B deeplearning/run_external.py` |
| Benchmark primer Otsu | `python -B deeplearning/run_otsu.py benchmark` |
| Evaluasi eksternal Otsu | `python -B deeplearning/run_otsu.py external_evaluation` |
| Buat gambar Otsu dari hasil tersimpan | `python -B deeplearning/plot_otsu.py benchmark` dan `python -B deeplearning/plot_otsu.py external_evaluation` |

Runner memakai jurnal hasil untuk melanjutkan run dengan konfigurasi yang sama. Untuk menghitung ulang dari awal, gunakan salinan kerja dengan direktori output baru atau pisahkan artefak yang sudah dipublikasikan sebelum menjalankan tahap tersebut. Evaluasi eksternal memakai artefak tuning primer yang sesuai. [Runner kontrol PCD tiga seed](deeplearning/run_pcd_control.py)

## Struktur repositori

```text
leaf-classification-ablation/
├── README.md
├── Dataset_Sekunder/                 # Manifest, sumber foto, dan lisensi
├── deeplearning/
│   ├── splits/                       # Outer dan inner CV primer
│   ├── run_benchmark.py
│   ├── run_external.py
│   ├── run_otsu.py
│   ├── plot_otsu.py
│   └── results/
│       ├── benchmark/                # DL primer, kondisi awal
│       ├── external_evaluation/      # Fit 60 primer, uji 150 sekunder
│       └── otsu/                      # Hasil primer dan eksternal Otsu
└── ablasi ML awal/
    ├── EDA_Ablasi_ML_Daun.ipynb
    ├── hasil_ml_full/                 # Baseline tanpa augmentasi
    │   ├── summary_108.csv
    │   ├── metrics_per_fold.csv
    │   ├── oof_predictions.csv
    │   ├── paired_deltas.csv
    │   ├── error_analysis_top4/
    │   ├── feature_importance_best_non_f4/
    │   └── ...
    └── hasil_ml_full_aug/             # Eksperimen dengan augmentasi train-only
        ├── augmentation_config.json
        ├── summary_108.csv
        ├── metrics_per_fold.csv
        ├── oof_predictions.csv
        ├── paired_deltas.csv
        ├── error_analysis_top4/
        ├── feature_importance_best_non_f4/
        └── ...
```

### Panduan artefak

Nama di bawah berlaku pada kedua folder hasil, kecuali jika disebutkan khusus.

| Berkas/folder | Isi |
|---|---|
| `summary_108.csv` | Ranking 108 kombinasi berdasarkan rata-rata macro-F1 |
| `metrics_per_fold.csv` | Skor, parameter, profil, dan timing setiap outer fold |
| `oof_predictions.csv` | Label sebenarnya dan prediksi OOF setiap foto |
| `paired_deltas.csv` | Selisih skor perlakuan pada fold, model, dan paket/input yang sama |
| `splits.csv` | Pembagian train dan evaluation per outer fold |
| `inner_tuning.csv`, `selected_profiles.json` | Hasil pencarian hyperparameter dan profil terpilih |
| `manifest_audit.csv` | Metadata citra, dimensi, brightness, blur, dan hash piksel |
| `qc_pcd.csv`, `feature_qc_profile0.csv` | Audit mask dan ketersediaan fitur |
| `error_analysis_top4/` | Analisis per kelas, confusion matrix, dan contoh kesalahan |
| `block_importance.csv`, `permutation_importance_by_fold.csv` | Importance blok dan fitur per fold |
| `feature_importance_best_non_f4/` | Analisis tambahan konfigurasi terbaik selain F4 |
| `hasil_ml_full/environment.json` | Catatan environment eksekusi baseline |

Path absolut pada artefak audit dan output notebook merupakan catatan mesin eksekusi lama. Path tersebut tidak diperlukan untuk membaca tabel hasil.

## Membaca dan menjalankan notebook

**Membaca hasil:** buka notebook di GitHub atau unduh dan buka secara lokal. Output eksekusi sudah tersimpan; CSV dan PNG dapat dibaca tanpa dataset atau training ulang.

**Menjalankan ulang:** sediakan dataset sendiri dengan struktur berikut pada root repo lokal. Dataset ini tidak termasuk dalam repositori.

```text
Dataset/
├── metadata.csv
├── Aglaonema/
│   └── aglaonema_1.jpg
└── <label lain>/
    └── <nama citra>
```

`metadata.csv` wajib memiliki kolom `image` dan `label`. Setiap citra dibaca dari `Dataset/<label>/<image>`. Jika memakai `plant_id`, kolom harus lengkap dan jumlah grup harus mencukupi untuk seluruh fold. Jika kolom `status_QC_label` tersedia, semua nilainya harus `approved`.

Siapkan environment Python dan dependensi:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install numpy pandas matplotlib pillow opencv-python scikit-learn threadpoolctl jupyterlab ipywidgets
python -m jupyterlab "ablasi ML awal/EDA_Ablasi_ML_Daun.ipynb"
```

Environment yang tercatat menggunakan Python 3.10.20, scikit-learn 1.7.2, dan OpenCV 5.0.0. Perintah instalasi di atas tidak mengunci versi; reproduksi angka yang sama belum dijamin untuk environment lain. Notebook menggunakan CPU dan tidak memerlukan PyTorch/CUDA.

Jalankan notebook dari root repo atau folder `ablasi ML awal`, lalu atur cell setup:

```python
SMOKE_TEST = False
SEED = 42
AUGMENTATION_ENABLED = True   # False untuk baseline
```

Dengan pengaturan ini, output ditulis ke `ablasi ML awal/hasil_ml_full_aug/`; baseline memakai `ablasi ML awal/hasil_ml_full/`. `SMOKE_TEST=True` memakai konfigurasi ringkas untuk memeriksa alur dan bukan benchmark final. Menjalankan ulang menulis ke folder hasil sesuai konfigurasi, sehingga simpan salinan hasil lama jika diperlukan.

Cell error analysis 7b dapat membaca CSV tersimpan tanpa training ulang. Cell feature importance 8 dan 8b membutuhkan objek model serta cache dari kernel training yang masih aktif. Beberapa narasi markdown lama pada notebook belum mengikuti penambahan augmentasi; konfigurasi aktif dan prosedur yang dijelaskan README ini mengacu pada kode serta CSV hasil, termasuk anchor `N–F8–SVM`.

## Batasan interpretasi

- Dataset kecil dan jumlah citra tiap kelas tidak seimbang. Satu kesalahan dapat memengaruhi skor kelas secara nyata.
- Hasil yang tersimpan memakai evaluasi tingkat foto karena `plant_id` tidak tersedia. Foto dari tanaman yang sama atau terkait dapat berada pada fold berbeda walaupun tidak identik secara piksel.
- `status_QC_label` belum tersedia pada data eksperimen; hasil masih provisional dan identitas botani belum divalidasi melalui protokol label independen.
- Ranking 108 kombinasi bersifat deskriptif. Skor konfigurasi yang dipilih setelah melihat OOF dapat optimistis; konfirmasi membutuhkan test set dari individu tanaman yang benar-benar terpisah.
- Augmentasi menambah variasi foto train, bukan jumlah individu tanaman independen.
- Segmentasi menggunakan prior warna, bukan deteksi semantik daun. Tidak tersedia ground truth mask untuk menyimpulkan IoU/Dice atau akurasi segmentasi.
- Timing berasal dari pengukuran komponen dan cache pada mesin eksperimen; tidak merepresentasikan benchmark deployment yang stabil.

Repositori ini menyimpan notebook dan artefak eksperimen, bukan model siap deployment. Notebook beserta output aslinya dipertahankan tanpa training ulang saat penyusunan dokumentasi ini.


## Error analysis ML tambahan dan reproduksi training

Folder yang sebelumnya bernama Notebook kini bernama **ablasi ML awal**. Dataset primer tidak disertakan. Referensi aktif notebook dan script pelaporan disesuaikan dengan nama folder baru.

- [Notebook lengkap dengan output eksekusi](ablasi%20ML%20awal/EDA_Ablasi_ML_Daun_Error_Analysis_Lengkap.ipynb)
- [Seluruh hasil run reproduksi](ablasi%20ML%20awal/reproduksi_github_f2473327_lengkap/)
- [Ringkasan pembandingan training](ablasi%20ML%20awal/reproduksi_github_f2473327_lengkap/reproduction_comparison.json)
- [Analisis tambahan, CSV dan figur](ablasi%20ML%20awal/reproduksi_github_f2473327_lengkap/error_analysis_tambahan/)
- [Snapshot referensi historis](ablasi%20ML%20awal/referensi_github_f2473327/)

Run reproduksi menghasilkan **6.480 prediksi identik**, skor per fold yang cocok, serta profil dan parameter terpilih yang sama dengan referensi. Ini kesamaan prediksi dan performa, bukan pembandingan bobot model lama.

Analisis tambahan mencakup perubahan prediksi per foto, jumlah SIFT dan perubahan warna, diagnosis mask, separabilitas fitur, kontrol foreground/background dengan Otsu dan K-Means, vocabulary bersama vs per kondisi, dan transfer antar kondisi. Kontrol diagnostik berparameter tetap tanpa augmentasi dibedakan dari benchmark nested CV. Tidak tersedia ground truth mask manual, sehingga background otomatis hanya proxy dan hasilnya bukan bukti kausal shortcut learning.

Untuk training ulang pada notebook lengkap:

    USE_GITHUB_ARTIFACTS = False
    REPRODUCE_GITHUB_RUN = True
    RUN_NEW_DIAGNOSTIC_MODELS = True

Pengaturan tersebut menjalankan training sesungguhnya. Untuk membaca artefak lama pilih USE_GITHUB_ARTIFACTS=True, REPRODUCE_GITHUB_RUN=False, dan nonaktifkan model diagnostik baru. Foto primer diperlukan untuk EDA, visualisasi, dan perhitungan fitur. Angka CSV dan output notebook dapat dibaca tanpa training.

Snapshot historis mempertahankan subfolder sumber_github/Notebook dan key manifest asli agar pemeriksaan hash terhadap commit referensi tetap sah. Ini bukan folder notebook aktif. Path mesin lama pada output tersimpan sengaja tidak diubah.
