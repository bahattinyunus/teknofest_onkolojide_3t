# 📝 TEKNOFEST 2026 — ÖDR (Ön Değerlendirme Raporu)
**Proje Adı:** GlioSight — Multimodal Onkoloji Karar Destek Sistemi  
**Takım ID:** [TAKIM ID BURAYA]  
**Kategori:** Onkolojide 3T

---

## 1. Proje Özeti
GlioSight, beyin kanseri (Glioblastoma) tanısı ve tedavisinde hekimlere yardımcı olan uçtan uca bir yapay zeka ekosistemidir. Sistem; 3D MRI segmentasyonu, moleküler sınıflandırma (WHO CNS 5), sürviyal tahmini, cerrahi/radyoterapi planlama ve algoloji takibi modüllerini tek bir çatı altında birleştirir.

## 2. Sorun / İhtiyaç
Glioblastoma, en agresif beyin tümörü türüdür. Tanı anında moleküler alt tiplerin belirlenmesi (MGMT, IDH) ve cerrahi marjinlerin hassas ayarlanması hayati önem taşır. Mevcut iş akışları parçalıdır ve manuel ölçümler hata payı barındırır.

## 3. Çözüm Önerisi
GlioSight, monolitik bir MRI analizinden öte, radyogenomik veriyi biyoteknolojik çıkarımlarla birleştirir. 3D U-Net tabanlı segmentasyon, %89+ Dice skoru ile tümör sınırlarını belirlerken; RANO kriterleri ile tedavi yanıtını, algoloji modülüyle ise hasta yaşam kalitesini takip eder.

## 4. Yenilikçi (İnovatif) Yönü
GlioSight'ı rakiplerinden ayıran en temel özellik, **Sovereignty Tier** mimarisi sayesinde sadece radyolojiye değil, aynı zamanda ilaç keşfi (Drug Discovery), neoantijen aşı önerisi ve OAR (Organ At Risk) koruma analizlerine de odaklanmasıdır.

## 5. Uygulanabilirlik
Proje, THS 4 seviyesinde olup, hastane PACS sistemlerine entegre edilebilir bir API ve Streamlit tabanlı kullanıcı arayüzü (Dashboard) ile sunulmaktadır.

## 6. Tahmini Maliyet ve Zaman Planı
- **Bulut Hesaplama (Eğitim):** 15.000 TL
- **Yazılım Geliştirme (6 Ay):** Takım içi
- **Toplam Bütçe:** ~25.000 TL (Fiziksel ekipman hariç)

---
*Bu rapor GlioSight v3.0 sistemi verileri baz alınarak taslak olarak oluşturulmuştur.*
