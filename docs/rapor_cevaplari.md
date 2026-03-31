# TEKNOFEST 2026 ONKOLOJİDE 3T YARIŞMASI
## PROJE ÖN DEĞERLENDİRME RAPORU (ÖDR)

> [!IMPORTANT]
> **Kullanım Notu:** Aşağıdaki metinler şartname kriterlerine (en fazla 2 sayfa metin, akademik dil, 12 punto Times New Roman uyumu) göre optimize edilmiştir. Doğrudan kopyalayıp sisteme yapıştırabilirsiniz.

---

### 1. BİYOTEKNOLOJİ ALANI
GlioSight projesi, onkoloji alanında en agresif ve mortalitesi yüksek beyin kanseri türü olan **Glioblastoma (GBM)** odaklı bütünleşik bir tanı ve karar destek sistemidir. Proje, TEKNOFEST 2026 şartnamesinde tanımlanan aşağıdaki kategorilerle doğrudan ilişkilidir:
- **Radyoloji ve Görüntüleme Teknolojileri:** Çok modlu (T1, T1ce, T2, FLAIR) MRI verilerinin yapay zekâ tabanlı 3B analizi.
- **Cerrahi Onkoloji Teknolojileri:** Tümör sınırlarının hassas belirlenmesi ve cerrahi güvenlik marjini simülasyonu.
- **Yapay Zekâ Destekli Yeni Nesil İlaç Geliştirme Çözümleri:** Radyogenomik analiz (MGMT metilasyon tahmini) yoluyla kemoterapiye yanıt öngörüsü.
- **Tıbbi Onkoloji:** Hastaya özel sağkalım (Survival) analizi ve risk skorlaması.

### 2. PROJE ÖZETİ
**Genel Hedef:** Glioblastoma hastalarında tanı, tedavi planlama ve prognoz takibi süreçlerini yapay zekâ yardımıyla dijitalize ederek hekim kararlarını veriye dayalı hale getirmektir.
**Temel Hipotez:** MRI hacimlerinden çıkarılan yüksek boyutlu radyomik öznitelikler, tümörün biyolojik agresifliğini ve genetik profilini (örn: MGMT metilasyonu) temsil eden dijital biyobelirteçlerdir.
**Önerilen Yaklaşım:** Proje; 3D U-Net mimarisi ile manuel hatadan arındırılmış 3B segmentasyon, Cox Proportional Hazards modelleri ile dinamik sağkalım tahmini ve Grad-CAM tabanlı "Açıklanabilir AI" (XAI) arayüzünü birleştiren hibrit bir ekosistem sunar.
**Proje Çıktıları:** (1) Otomatik 3B Segmentasyon Modülü, (2) Sağkalım ve Risk Analiz Motoru, (3) Radyogenomik Tahmin Aracı, (4) Cerrahi Planlama Asistanı ve (5) PACS entegrasyonu için REST API mimarisi.

### 3. SORUN TANIMI
**Hedef Sorun:** Beyin kanserlerinin (özellikle Glioblastoma) klinik yönetimindeki belirsizlikler ve operasyonel gecikmeler hedeflenmektedir.
**Giderilmesi Hedeflenen Zorluklar:**
- **Analiz Değişkenliği:** Radyologlar arası manuel segmentasyon farklılıkları tedavi planlamasında standart hataya yol açmaktadır.
- **Klinik Boşluk:** Mevcut radyolojik yazılımlar sadece "görüntüleme" odaklı olup, hastanın yaşama şansı (OS) veya genetik mutasyon durumu hakkında sayısal bilgi sunmamaktadır.
- **Cerrahi Belirsizlik:** Tümörün infiltrasyon (sızma) alanlarının çıplak gözle ayırt edilememesi, nüks riskini artıran yetersiz rezeksiyona veya fonksiyon kaybına yol açan aşırı cerrahi marjinlere sebep olmaktadır.

### 4. ÇÖZÜM
GlioSight, belirtilen sorunları **"Hassas Onkoloji Motoru"** ile çözmeyi planlar:
- **Teknoloji ve Yöntem:** Pipeline; NIfTI formatındaki MRI verilerini normalize eder, monai framework'ünde eğitilmiş 3D U-Net ile tümör çekirdeği ve ödemli dokuyu 1mm³ hassasiyetle ayırır.
- **Özgünlük ve Uygulanabilirlik:** Çözüm, sadece segmentasyon yapmakla kalmaz; segmentasyon sonuçlarını bir girdi olarak kullanarak cerrahi bir "güvenlik marjini" (5-10mm) haritası çıkarır.
- **Klinik Katkı:** Hekimin önüne cerrahi öncesi (pre-op) hem genetik profil (MGMT) hem de muhtemel sağkalım süresini koyarak, tedavinin agresiflik düzeyinin (palyatif vs. küratif) belirlenmesine somut katkı sunar.

### 5. YENİLİKÇİ YÖNÜ VE ÖZGÜN DEĞERİ
**Teknolojik Yenilikçilik:** GlioSight, piyasadaki "Kapalı Kutu" (Black Box) AI modellerinin aksine **Açıklanabilir AI (XAI)** özelliğine sahiptir. Grad-CAM ısı haritaları sayesinde, modelin neden bir dokuyu nüksetme riski yüksek olarak gördüğünü cerraha anatomik olarak açıklar.
**Literatürden Farkı:** Klasik yöntemler tek modalite veya 2B kesitler üzerinde çalışırken, GlioSight multimodal (T1+T2+FLAIR) hacimsel füzyon yaparak tümörün 3B geometrisini ve doku dokusunu (texture) derinlemesine analiz eder.
**Alana Katkı:** Proje, biyoteknoloji ve yazılım mühendisliğini "Radyogenomik" potasında eriterek Türkiye'de dijital onkoloji alanında yerli ve özgün bir karar destek prototipi oluşturmaktadır.

### 6. TEKNOLOJİ HAZIRLIK SEVİYESİ (THS)
**Konumlandırma:** Proje şu anda **THS 3 (Konsept Kanıtlanmış)** seviyesindedir.
**Dayanak:** 
- Laboratuvar ortamında uluslararası altın standart kabul edilen **BraTS (Brain Tumor Segmentation Challenge)** veri setleri kullanılarak konseptin doğruluğu test edilmiştir.
- Algoritmaların temel bileşenleri (3D U-Net segmentasyonu ve Cox-Regresyon) analitik olarak simüle edilmiş ve performans metrikleri (Dice Score, C-Index) elde edilmiştir.
- Henüz gerçek bir klinik operasyonel ortamda (hastane PACS ağı) canlı veriyle pilot uygulama (THS 4+) yapılmamış olup, tasarım ve laboratuvar validasyonu tamamlanmıştır.

---

### 📚 KAYNAKÇA (Örnek Akademik Format)
1. Menze, B. H., et al. (2015). "The Multimodal Brain Tumor Segmentation Benchmark (BRATS)". IEEE TMI.
2. Bakas, S., et al. (2018). "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge". arXiv.
3. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". ICCV.

---
**NOT:** Raporu KYS sistemine yüklerken; kapak sayfası dahil en fazla 4 sayfa olmasına ve metinlerin 2 sayfayı geçmemesine dikkat ediniz. Proje kodları üzerinden üretilen 3B segmentasyon ve radyomik analiz görsellerini "Görseller" kısmına eklemeyi unutmayınız.
