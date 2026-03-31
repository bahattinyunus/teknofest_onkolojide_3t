![Onkolojide 3T Banner](assets/oncology_3t_premium_banner.png)

# 🔬 GlioSight — Multimodal MRI AI Platform (v2.0)
> **TEKNOFEST 2026: Onkolojide 3T Yarışması — Tam Kapsamlı (8/8) Klinik Karar Destek Sistemi**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-blue)](https://monai.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Bu depo, **TEKNOFEST 2026 Onkolojide 3T** yarışması kapsamında geliştirilen, Glioblastoma (GBM) hastaları için uçtan uca **Tanı, 3B Segmentasyon, Sağkalım, Radyogenomik, Radyasyon Planlama, Dijital Patoloji ve Hassas Tıp** ekosistemini içermektedir.

---

## 🎯 1. Yarışma Amacı ve Kapsamı
Yarışma, onkoloji alanında **bireyselleştirilmiş (personalised)** ve **hassas (precision)** tedavi teknolojilerinin geliştirilmesini, genetik mühendisliği, biyoteknoloji ve yapay zekâ odaklı klinik etki potansiyeli yüksek çözümler üretmeyi amaçlar.

### 🧠 Hedef Hastalık: Beyin Kanseri
TEKNOFEST 2026 döneminde tüm çalışmalar **Beyin Kanseri** türleri üzerine yoğunlaşmaktadır.

---

## 🚀 2. GlioSight Temel Özellikler (Core Features)

GlioSight, radyologlar ve onkologlar için 4 ana modülden oluşan bütünleşik bir karar destek sistemi sunar:

### 🧩 2.1. 3B Otomatik Segmentasyon (Segmentation)
*   **Mimari:** Residual-Dilation 3D U-Net (MONAI tabanlı).
*   **Kapasite:** T1, T1ce, T2 ve FLAIR modalitelerini senkronize işleyerek **NCR/NET** (Nekrotik), **ED** (Ödem) ve **ET** (Kontrast Tutan) bölgelerini ayrıştırır.

### 📈 2.2. Radyomik Sağkalım Analizi (Survival)
*   **Yöntem:** PyRadiomics öznitelikleri üzerinden **Cox Proportional Hazards** ve **XGBoost Survival** analizi.
*   **Çıktı:** Hastaya özel Kaplan-Meier sürviyal eğrileri ve sayısal risk skorlaması.

### 💡 2.3. Açıklanabilir AI (Explainable AI - XAI)
*   **Grad-CAM:** Modelin hangi tümör bölgesine odaklanarak risk tahmini yaptığını gösteren 3B ısı haritaları.
*   **SHAP:** En belirleyici 15 radyomik özelliğin (örn: Elongation, Sphericity) risk skoru üzerindeki etkisini görselleştirir.

### 🩺 2.4. Cerrahi & Radyasyon Planlama (Surgical/Radiation)
*   **Surgical:** 5mm-15mm dinamik cerrahi marjin ve "Safety Score" puanlaması.
*   **Radiation:** Otomatik **CTV (20mm)** ve **PTV (3mm)** hedef hacim üretimi (ESTRO Standartları).

### 🧪 2.5. Dijital Patoloji & Hassas Tıp (Biotech)
*   **Pathology:** MRI verisinden emüle edilmiş **Ki-67 proliferasyon indeksi** ve mitoz sayısı tahmini.
*   **Precision Medicine:** MGMT durumuna bağlı **Temozolomide (TMZ)** duyarlılık skoru ve tedavi yolu önerisi.

---

## 📅 3. Yarışma Takvimi (2025-2026)
| Aşama | Tarih |
| :--- | :--- |
| **Yarışma Son Başvuru Tarihi** | *İlan edilecektir* |
| **Proje Ön Değerlendirme Raporu** | *İlan edilecektir* |
| **Yarı Final Raporu** | *İlan edilecektir* |
| **Final Raporu & Video Sunumu** | *İlan edilecektir* |
| **TEKNOFEST Final Etkinliği** | 2026 |

---

## 📊 4. Puanlama ve Değerlendirme Sistemi

### 4.1. Değerlendirme Aşamaları
1.  **Ön Değerlendirme Raporu:** Projenin temel amacı, yenilikçiliği ve uygulanabilirliği.
2.  **Yarı Final Raporu:** Teknik detaylar, analiz sonuçları ve klinik potansiyel.
3.  **Final Sunumu:** Sözlü sunum ve poster sunumu üzerinden gerçekleştirilir.

### 4.2. Puan Dağılımı
*   **Ön Değerlendirme Raporu:** %10
*   **Yarı Final Raporu:** %25
*   **Final Raporu:** %35
*   **Final Sunumu:** %30

---

## 🏆 5. Ödüller ve Teşvikler
*   **Birincilik:** 200.000 TL
*   **İkincilik:** 150.000 TL
*   **Üçüncülük:** 120.000 TL
*   **Özel Ödüller:** En İyi Sunum, Gelecek Vadeden Yenilikçi Proje, En İyi Yapay Zeka Odaklı Proje.

---

## 📂 6. Proje Yapısı
```text
├── data/               # Veri setleri ve ön işleme betikleri
├── models/             # AI ve hesaplamalı biyoloji modelleri
├── src/                # Ana algoritma ve uygulama kodları
├── experiments/        # Test sonuçları ve analiz metrikleri
├── docs/               # Yarışma raporları ve sunum dosyaları
└── assets/             # Görsel materyaller ve diyagramlar
```

---

## ⚖️ 7. Etik Kurallar ve Fikri Mülkiyet
*   Tüm projeler için **Etik Kurul Onay Belgesi** (gerekli ise) ibraz edilmelidir.
*   Fikri mülkiyet hakları, yarışmacı ve Cansağlığı Vakfı arasında yapılacak protokoller çerçevesinde düzenlenir.
*   Projelerin özgün olması ve akademik dürüstlük ilkelerine bağlı kalınması zorunludur.

---

## 🔍 8. Rakip Analizi ve Referanslar

Başarılı bir proje geliştirmek için geçmiş yılların kazananları ve uluslararası standartlar analiz edilmiştir.

### 🏆 Geçmiş TEKNOFEST Kazananları (2025 - Akciğer Kanseri Odağı)
*   **Dockminds (1.lik):** Çok modlu yapay zeka ve protein dinamiği destekli ilaç keşfi. Allosterik inhibitör tasarımı ve in vitro doğrulama ile öne çıkmışlardır.
*   **CalpaCure (3.lük):** Biomarker keşfi ve shRNA tabanlı genetik tedavi yaklaşımları.

### 🌐 Uluslararası Yarışmalar ve Başvuru Kaynakları
Beyin tümörü analizi (segmentasyon ve sınıflandırma) alanındaki dünya standartları:

*   **MICCAI BraTS Challenge:** Beyin tümörü segmentasyonu ve hayatta kalma süresi tahmini üzerine en prestijli yarışmadır.
*   **RSNA Glioblastoma Classification:** Glioblastoma alt tiplerinin MRI üzerinden sınıflandırılması.

### 💻 Açık Kaynaklı Kod Referansları
Projemizde temel alınabilecek yüksek performanslı GitHub projeleri:

1.  **[BraTS 2021 Implementation](https://github.com/younesbelkada/BraTS_2021):** Glioblastoma segmentasyonu için 3D U-Net mimarisi.
2.  **[Trusted BraTS (TBraTS)](https://github.com/Cocofeat/TBraTS):** Belirsizlik kestirimi yapan güvenilir segmentasyon modelleri.
3.  **[Multimodal MP-MRI Segmentation](https://github.com/mfaizan-ai/Brain-Tumors-Segmentation):** 2023 BraTS verisiyle güncel segmentasyon boru hattı.
4.  **[3D U-Net survival prediction](https://github.com/woodywff/brats_2019):** BraTS 2019 dünya 3.lüğü çözümü.

---

## 🚀 9. Kullanım ve Çalıştırma

GlioSight sistemini yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin:

### 9.1. Kurulum
```bash
git clone https://github.com/bahattinyunus/teknofest_onkolojide_3t.git
cd teknofest_onkolojide_3t
pip install -r requirements.txt
```

### 9.2. Uçtan Uca Çıkarım (Segmentasyon + Sağkalım)
Tek bir hasta dizini üzerinde tüm analizleri çalıştırmak için `src/main.py` motorunu kullanabilirsiniz:

```bash
python src/main.py --data_dir data/raw/BraTS2021_00001 --output_dir results/
```

**Parametreler:**
- `--data_dir`: MRI modalitelerinin (T1, T1ce, T2, FLAIR) bulunduğu hasta klasörü.
- `--output_dir`: Sonuç görsellerinin ve raporun kaydedileceği dizin.
- `--seg_ckpt`: (Opsiyonel) Eğitilmiş 3D U-Net model ağırlıkları yolu.

### 9.3. Çıktılar
İşlem tamamlandığında `results/` klasöründe şu dosyalar oluşacaktır:
- `segmentation.png`: MRI kesitleri ve renkli tümör maskesi overlay görüntüsü.
- `report.txt`: Risk skoru, sağkalım düzeyi ve hacimsel tümör istatistiklerini içeren metin raporu.

---

*Bu doküman TEKNOFEST 2026 Onkolojide 3T yarışma şartnamesi baz alınarak oluşturulmuştur.*
