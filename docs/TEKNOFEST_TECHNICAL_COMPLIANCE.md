# 🛡️ GlioSight — TEKNOFEST 2026 Teknik Uyumluluk Matrisi

Bu doküman, **GlioSight** multimodal AI platformunun **Onkolojide 3T Yarışması** şartnamesinde belirtilen teknik kategorilerle olan uyumluluğunu detaylandırmaktadır.

---

## 🏗️ 1. Şartname Kapsam Eşleşmesi

| Şartname Kategorisi | GlioSight Modülü | Teknik Yetkinlik | Dosya Referansı |
| :--- | :--- | :--- | :--- |
| **Cat 7: Radyasyon Onkolojisi** | `RadiationPlanner` | ESTRO standartlarında otomatik CTV (20mm) ve PTV (3mm) hacim üretimi. | `src/utils/radiation_planner.py` |
| **Cat 8: Patoloji** | `PathologyEmulator` | MRI tabanlı Ki-67 proliferasyon indeksi ve selülarite tahmini (Dijital Patoloji Emülasyonu). | `src/utils/pathology_emulator.py` |
| **Cat 9: Radyoloji & Görüntüleme** | `SegmentationPipeline` | 4 modaliteli (T1, T1ce, T2, FLAIR) 3D Residual U-Net ile tümör segmentasyonu. | `src/inference/segmentation_pipeline.py` |
| **Cat 9: Radyomik & Radyogenomik** | `RadiogenomicPipeline` | WHO CNS 5 uyumlu MGMT, IDH ve 1p/19q moleküler marker tahmini. | `src/inference/radiogenomic_pipeline.py` |
| **Cat 9: Tedavi Yanıt Analizi** | `RANOModule` | Hacimsel değişim üzerinden RANO (Response Assessment in Neuro-Oncology) kriterleri ile yanıt sınıflandırma. | `src/utils/rano_criteria.py` |
| **Cat 11: Cerrahi Onkoloji** | `SurgicalPlanner` | 3D navigasyon destekli cerrahi marjin (Safety Margin) hesaplama ve eleştirel alan yakınlık uyarısı. | `src/utils/surgical_planner.py` |
| **Cat 12: Biyomedikal Cihaz** | `GlioSight Dashboard` | Hekimler için Streamlit tabanlı interaktif karar destek ve biyoizleme arayüzü. | `src/api/dashboard.py` |

---

## 🧬 2. Klinik ve Bilimsel Standartlar (WHO CNS 5)

GlioSight, **Dünya Sağlık Örgütü (WHO) 5. Sürüm Santral Sinir Sistemi Tümör Sınıflandırması** kriterlerini temel alır:

1.  **IDH Mutasyonu:** Glioblastoma (IDH-wildtype) ve Astrocytoma (IDH-mutant) ayrımı için T2-FLAIR mismatch analizi.
2.  **1p/19q Kodelesyonu:** Oligodendroglioma tanısı için morfolojik ve radyogenomik özellik çıkarımı.
3.  **MGMT Metilasyonu:** Temozolomide (TMZ) kemoterapi yanıtı öngörüsü.

---

## 📊 3. Teknik Hazırlık Seviyesi (THS)

Projemiz, şartnamede belirtilen **THS 4 (Laboratuvar ortamında prototip)** seviyesine tam uyumludur. Algoritmalar laboratuvar setlerinde doğrulanmış, bütünleşik bir yazılım ekosistemi haline getirilmiştir.

---

## ⚖️ 4. Etik ve KVKK Uyumluluğu

*   **Veri:** Tüm eğitim ve test süreçlerinde anonimleştirilmiş (de-identified) BraTS veri setleri kullanılmıştır.
*   **KVKK:** `docs/form_cevaplari.md` dosyasında kişisel verilerin korunması ve etik ilkeler açıkça beyan edilmiştir.

---
*Bu matris jüri değerlendirme raporu (ÖDR/YFR) için temel teşkil eder.*
