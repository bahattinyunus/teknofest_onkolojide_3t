# TEKNOFEST 2026 ONKOLOJİDE 3T YARIŞMASI
## PROJE ÖN DEĞERLENDİRME RAPORU (ÖDR)

**Proje Adı:** GlioSight — Multimodal MRI ve Yapay Zekâ Destekli Glioblastoma Tanı, Segmentasyon ve Sağkalım Analiz Sistemi

**Takım Adı:** [Takım Adınız Buraya]
**Başvuru ID:** [KYS ID Buraya]

---

### 1. PROJE ÖZETİ (Özet)
GlioSight projesi, beyin kanserlerinin en agresif türü olan Glioblastoma (GBM) hastalarında tanı ve tedavi süreçlerini optimize etmek amacıyla geliştirilmiş uçtan uca bir yapay zekâ platformudur. Sistem, multimodal MRI verilerini (T1, T1ce, T2, FLAIR) kullanarak; tümör alt bölgelerinin (Geniş Tümör, Tümör Çekirdeği, Kontrast Tutan Bölge) 3B segmentasyonunu gerçekleştirmekte ve bu segmentasyonlardan çıkarılan radyomik özelliklerle hastanın sağkalım (OS) süresini tahmin etmektedir. Projede, derin öğrenme tabanlı 3D U-Net mimarisi ve Cox Proportional Hazards istatistiksel modelleri entegre edilerek, hekime karar destek sağlayacak hassas bir dijital onkoloji çözümü sunulmaktadır.

### 2. PROBLEM TANIMI VE ÇÖZÜM YAKLAŞIMI
**Problem:** Glioblastoma tanısında, tümörün heterojen yapısı nedeniyle radyolojik görüntüler üzerinden manuel segmentasyon yapılması zaman alıcıdır ve gözlemciler arası değişkenliğe açıktır. Ayrıca, hastanın tedaviye yanıtının ve sağkalım beklentisinin sadece görsel analizle tahmin edilmesi, kişiselleştirilmiş tedavi planlaması (precision oncology) için yetersiz kalmaktadır.

**Çözüm:** GlioSight, bu problemi üç aşamalı bir yaklaşımla çözmektedir:
1. **Otomatik Segmentasyon:** Manuel segmentasyon yükünü ortadan kaldıran ve BraTS standartlarında yüksek doğuluk sunan derin öğrenme modeli.
2. **Sağkalım Analizi:** Tümör hacmi, sferisite ve yoğunluk gibi radyomik verileri klinik parametrelerle birleştirerek risk katmanlaması yapan AI motoru.
3. **Klinik Karar Destek:** Hekime metastatik risk ve prognoz hakkında sayısal veriler sunan görsel raporlama sistemi.

### 3. YENİLİKÇİ-ÖZGÜN YÖNÜ
Projenin mevcut çözümleren ayıran temel özellikleri şunlardır:
- **Hibrit Mimari:** Sadece görüntü işleme değil, radyomik veriler üzerinden sağkalım regresyonu yapan (Cox PH + XGBoost Survival) entegre bir yapı sunması.
- **Yüksek Çözünürlüklü 3B Analiz:** Kesit bazlı (2B) değil, hacimsel (3B) veriyi tüm derinliğiyle işleyerek tümörün gerçek geometrisini analiz etmesi.
- **Açıklanabilir AI (XAI):** SHAP analizi ile modelin hangi radyomik özelliğe (örn: ET/TC oranı) dayanarak risk tahmini yaptığını gösteren şeffaf bir yapı.

### 4. TEKNİK MİMARİ VE YÖNTEM
1. **Veri Hazırlama:** NIfTI formatındaki MRI görüntüleri Z-Score normalizasyonu ve bias field correction işlemlerinden geçirilir.
2. **Derin Öğrenme Modeli:** BraTS 2021 veri setiyle eğitilen, Dice + Focal Loss fonksiyonlarına dayalı 3D U-Net mimarisi (Residual-Dilation blokları ile güçlendirilmiş).
3. **Özellik Çıkarımı:** PyRadiomics tabanlı hacimsel, yoğunluksal ve doku (texture) özellikleri.
4. **Tahmin:** Sağkalım süresi için Cox Regresyon ve risk skorlaması.
5. **Teknoloji Yığını:** Python, PyTorch, MONAI, Lifelines, Matplotlib.

### 5. BEKLENEN ÇIKTILAR VE ETKİ
- **Tanı Süresi:** Radyologların tümör segmentasyon süresinin %90 oranında kısaltılması.
- **Tedavi Planı:** Kişiselleştirilmiş tedavi süreçlerinde sağkalım tahmini ile daha agresif veya palyatif yaklaşımlara karar verme sürecinde %15-20 artış gösteren prognostik doğruluk.
- **Yaygın Etki:** Yazılımın hastane PACS sistemlerine entegre edilebilir bir API yapısıyla sunulması.

---
**Teslim Tarihi:** 31 Mart 2026
**Teknoloji Hazırlık Seviyesi (THS):** 3 (Konsept kanıtlanmış, laboratuvar ortamında validasyon süreci devam etmektedir).
