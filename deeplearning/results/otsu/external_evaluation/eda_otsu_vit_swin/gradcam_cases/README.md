# Grad-CAM berpasangan untuk kasus Otsu

Analisis ini memakai foto sekunder yang sama, dengan model pretrained dan augmentasi OFF. Model dilatih ulang pada 60 foto primer menggunakan recipe eksternal yang tercatat. Pada kondisi `S-O`, Otsu hanya diterapkan saat training. Enam refit menghasilkan label prediksi yang cocok dengan arsip pada **900 dari 900 evaluasi**. [Pemeriksaan refit](refits.csv) · [Konfigurasi](run_config.json)

## Lima foto yang membaik pada ViT dan Swin

Keempat peta pada satu foto memakai kelas asli sebagai target Grad-CAM yang sama. Skor target adalah nilai softmax dari inferensi. Warna peta dinormalisasi per panel untuk membaca posisi respons spasial. Perbandingan besarnya skor kelas memakai angka softmax.

| Foto | Label asli | ViT `N` → `S-O` | Swin `N` → `S-O` |
|---|---|---|---|
| [iNat 19219491](gain_vit_swin_19219491.png) | Kamboja jepang | Nangka → Kamboja | Paku → Kamboja |
| [iNat 245161059](gain_vit_swin_245161059.png) | Kamboja jepang | Paku → Kamboja | Paku → Kamboja |
| [iNat 418438754](gain_vit_swin_418438754.png) | Aglaonema | Lidah metua → Aglaonema | Lidah metua → Aglaonema |
| [iNat 245774771](gain_vit_swin_245774771.png) | Peperomium semangka | Paku → Peperomia | Paku → Peperomia |
| [iNat 471692010](gain_vit_swin_471692010.png) | Daun nangka | Kamboja → Nangka | Kamboja → Nangka |

Pada contoh iNat 245161059, skor kamboja naik dari **0,035 ke 0,956** pada ViT dan dari **0,068 ke 0,607** pada Swin. Peta setelah S-O masih memberi respons di beberapa bagian tanaman dan latar. [Lihat contoh kamboja](gain_vit_swin_245161059.png)

## Kenapa skor paku turun pada ResNet-50?

**Pada kondisi `S-O` tanpa augmentasi, ResNet berhenti memilih kelas paku.** Model `N` memberi label paku pada 54 dari 150 foto, termasuk seluruh 25 foto paku. Model `S-O` memberi label paku pada **0 dari 150**. Dari 25 foto paku, 21 berpindah ke Peperomium semangka, 3 ke Lidah metua, dan 1 ke Kamboja jepang. Median skor untuk kelas paku pada 25 foto tersebut turun dari **0,968 ke 0,042**. [Prediksi dan skor kelas paku](prediksi_paku_resnet.csv)

| Contoh paku | Prediksi `S-O` | Skor kelas paku `N` → `S-O` |
|---|---|---:|
| [iNat 721118477](loss_resnet_paku_721118477.png) | Peperomium semangka | 0,984 → 0,011 |
| [iNat 728790409](loss_resnet_paku_728790409.png) | Peperomium semangka | 0,991 → 0,029 |
| [iNat 715480225](loss_resnet_paku_715480225.png) | Lidah metua | 0,970 → 0,097 |
| [iNat 757327216](loss_resnet_paku_757327216.png) | Kamboja jepang | 0,557 → 0,040 |

Dua peta pertama di tiap contoh ResNet menargetkan paku pada kedua kondisi. Peta terakhir menargetkan label salah yang dipilih model `S-O`. Perbandingan ini menunjukkan lokasi respons untuk kelas paku dan lokasi respons untuk label penggantinya. [Target dan skor setiap peta](skor_dan_target_cam.csv) · [Atribusi foto](atribusi_foto.csv)

Pada iNat 721118477, peta target paku dari N menonjolkan rumpun frond di tengah. Setelah S-O, area paling menonjol untuk target paku bergeser ke sisi kiri foto, sedangkan area yang menonjol pada peta target Peperomia tetap berada di rumpun tanaman. Pada iNat 757327216, target paku masih memberi respons di sebagian frond walaupun skor kelasnya jatuh. Keduanya memperlihatkan perubahan respons yang berbeda pada foto uji yang sama. [Kasus Peperomia](loss_resnet_paku_721118477.png) ? [Kasus Kamboja](loss_resnet_paku_757327216.png)

**Yang terlihat pada foto latih.** Mask `S-O` mempertahankan median **45,6%** dari seluruh piksel pada 14 foto primer paku, dengan rentang **23,8?63,2%**. Pada tiga contoh di panel, bentuk utama daun paku masih terlihat. Perubahan yang jelas ada pada konteks di sekitarnya, termasuk batang pohon, tanah, dan tanaman lain. Seluruh foto uji masuk ke model dengan persiapan tensor yang sama. Perubahan konteks saat training bisa memengaruhi isyarat yang dipelajari model, tetapi panel ini belum menjelaskan penyebab turunnya skor kelas paku. [Panel mask latih paku](resnet_paku_mask_latih.png) ? [Audit mask primer](../../../benchmark/mask_audit.csv)

**Cakupan temuan.** Pada foto paku yang sama, ViT berubah dari 22/25 ke 19/25 dan Swin dari 24/25 ke 25/25. ResNet dengan `S-O` dan augmentasi ON memperoleh 15/25. Untuk ResNet tanpa augmentasi, `N` dan `S-O` memakai LR `1e-4` dengan masing-masing 11 dan 13 epoch. Peta Grad-CAM memberi petunjuk spasial. Menetapkan penyebab penurunan memerlukan kontrol durasi, seed, dan anotasi daun.

Kode regenerasi tersedia di [gradcam_otsu_cases.py](../../../../../gradcam_otsu_cases.py). Pipeline membutuhkan data primer dan sekunder, CUDA, dan environment benchmark.
