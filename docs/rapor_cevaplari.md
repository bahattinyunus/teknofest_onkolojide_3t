# TEKNOFEST 2026 ONKOLOJİDE 3T YARIŞMASI
## PROJE ÖN DEĞERLENDİRME RAPORU (ÖDR) — GENİŞLETİLMİŞ VERSİYON

---

### 1. BİYOTEKNOLOJİ ALANI
**GlioSight** projesi, beyin kanserlerinin en progresif ve tedaviye dirençli türü olan **Glioblastoma Multiforme (GBM)** hastaları için geliştirilmiş yüksek sadakatli (High-Fidelity) bir dijital onkoloji ekosistemidir. Proje, TEKNOFEST 2026 şartnamesinde yer alan aşağıdaki biyoteknolojik kategorileri uçtan uca bir mimaride birleştirir:
- **Radyoloji ve Görüntüleme Teknolojileri:** T1, T1-kontrastlı (T1ce), T2 ve FLAIR modalitelerini 3 boyutlu uzayda voxel bazlı işleyerek tümörün spasyal heterojenliğini analiz eder.
- **Cerrahi Onkoloji Teknolojileri:** Tümörün "infiltrasyon" (sızma) zonlarını otomatik tespit ederek cerraha 3B "Güvenlik Koridoru" ve dinamik cerrahi marjin (5mm, 10mm, 15mm) önerileri sunar.
- **Yapay Zekâ Destekli Yeni Nesil İlaç Geliştirme Çözümleri:** Biyobelirteç tahmini (MGMT Promoter Metilasyonu) yaparak kişiselleştirilmiş kemoterapi protokollerinin (Precision Medicine) oluşturulmasına imkan tanır.
- **Tıbbi Onkoloji:** Radyomik öznitelikleri (Radiomic Features) klinik parametrelerle birleştirerek hastaya özgü sağkalım (Survival) eğrileri üretir.

### 2. PROJE ÖZETİ
**Genel Hedef ve İçerik:** GlioSight, beyin kanseri tanı ve tedavi sürecindeki "insan hatası" ve "subjektif değerlendirme" faktörlerini minimize etmeyi amaçlar. Sistem, raw MRI verisini alıp saniyeler içinde 3B segmentasyon, radyogenomik biyobelirteç tahmini ve cerrahi planlama raporu üreten entegre bir yapay zekâ motorudur.
**Temel Hipotez:** Tıbbi görüntüler, insan gözünün ayırt edemediği "gizli" (latent) doku dokusu bilgileri barındırır. Bu doku bilgileri (Digital Biopsy), tümörün genetik mutasyon durumu ve hastanın toplam yaşam süresi (OS) ile matematiksel olarak ilişkilendirilebilir.
**Yaklaşım:** Projede derin öğrenme mimarisi olarak **3D Residual U-Net (ResUNet)** kullanılmıştır. Segmentasyon sonuçları otomatik olarak hacimsel (volumetric), morfolojik (sphericity, elongation) ve yoğunluksal (histogram-based) özniteliklere dönüştürülüp **Cox Proportional Hazards Inference** motoruna beslenmektedir.
**Proje Çıktıları:** (1) Hekime sunulan 3B etkileşimli segmentasyon maskesi, (2) Sayısal risk skorlaması ve sağkalım süresi tahmini, (3) MGMT metilasyon ihtimalini gösteren radyogenomik rapor ve (4) PACS entegrasyonuna uygun RESTful API mimarisi.

### 3. SORUN TANIMI
**Hedef Sorun:** Glioblastoma tanısında "standart olmayan" analiz süreçleri ve cerrahi öncesi (pre-operative) prognostik belirsizlik hedeflenmektedir.
**Giderilmesi Hedeflenen Zorluklar:**
- **Heterojen Yapı:** GBM tümörleri; nekrotik çekirdek, kontrast tutan aktif doku ve ödemli sızıntı alanlarından oluşur. Bu bölgelerin manuel olarak ayrıştırılması (segmentasyon) hem zaman alıcıdır hem de radyologlar arasında Dice skoru bazında %15-20 sapmaya neden olur.
- **Tanısal Mesafe:** Mevcut sistemler sadece "tümör var/yok" veya "boyut nedir?" sorularına yanıt verirken; tümörün genetik profili (örn: MGMT) ancak haftalar süren invaziv biyopsi sonuçları ile öğrenilebilmektedir. Bu durum tedavi başlangıcında kritik zaman kayıplarına yol açar.
- **Rezeksiyon Riski:** Cerrahi sırasında tümör sınırlarının mikroskobik düzeyde tam belirlenememesi, ya sağlam dokunun hasar görmesine ya da tümörün içeride kalmasına (Subtotal Resection) neden olarak nüks riskini %80 artırır.

### 4. ÇÖZÜM
GlioSight, belirtilen yapısal sorunları **"Hassas Onkoloji ve Biyoinformatik Standartları"** doğrultusunda şu yöntemlerle çözer:
- **Teknolojiler:** PyTorch ve MONAI framework’leri üzerinde geliştirilen 3D U-Net motoru, braTS standartlarında eğitilmiş olup "Sliding Window Inference" yöntemiyle yüksek çözünürlüklü MRI’ları bütüncül işler.
- **Yöntem ve Yaklaşım:** Proje, biyopsiden önce "Dijital Biyopsi" (Radiomics) yaparak MGMT metilasyon durumunu %80+ AUC doğrulukla tahmin eder. Bu, onkoloğun cerrahi öncesinde hastanın Temozolomid (TMZ) tedavisine duyarlılığını bilmesini sağlar.
- **Özgünlük:** Çözümün en ayırıcı özelliği, segmentasyon maskesi üzerinden otomatik olarak **Cerrahi Marjin Simülasyonu** gerçekleştirmesidir. Morfoloji analizi yaparak tümörün en riskli infiltrasyon yönlerini belirler ve cerraha "önerilen rezeksiyon hacmi" simülasyonunu sunar.
- **Uygulanabilirlik:** Sistem tamamen modülerdir. Hastanelerdeki PACS (Picture Archiving and Communication System) sistemlerine FastAPI katmanıyla doğrudan bağlanarak radyolog ekranına bir "Karar Destek Widget'ı" olarak eklenebilir.

### 5. YENİLİKÇİ YÖNÜ VE ÖZGÜN DEĞERİ
**Teknolojik Yenilikçilik (XAI):** GlioSight, tıbbi yapay zekânın en büyük sorunu olan "güven" problemini **Açıklanabilir AI (Explainable AI - Grad-CAM)** ile aşmaktadır. Model, bir bölgeyi neden yüksek riskli işaretlediğini gradyan bazlı ısı haritalarıyla cerraha göstererek "karar gerekçelendirmesi" yapar.
**Bilimsel Özgünlük:** Literatürde genellikle sadece görüntü işleme veya sadece istatistiksel sürviyal analizi yapılırken; GlioSight'ta segmentasyon, radyogenomik ve cerrahi planlama **tek bir hibrit pipeline** altında birleşmiştir. "End-to-End Clinical Insight" sunan bu yapı, esinlenmenin ötesinde özgün bir mühendislik tasarımıdır.
**Alana Katkılar:** Türkiye'nin "Onkolojide Yerli Yazılım" vizyonuna uygun olarak, biyomedikal veri işleme süreçlerini yapay zekâ otoritesiyle birleştirir. Pahalı ve zaman alan genetik testlere (NGS vb.) dijital bir alternatif/ön-test sunarak stratejik bir ekonomik değer yaratır.

### 6. TEKNOLOJİ HAZIRLIK SEVİYESİ (THS)
**Mevcut Seviye:** **THS 3 (Konsept Kanıtlanmış / Analitik ve Deneysel Doğrulama)**
**THS Seviyesinin Dayanağı:** 
- **Veri Doğrulaması:** Algoritmalar, 500'den fazla hastayı içeren BraTS 2021 ve 2023 veri setleri (Benchmark) üzerinde başarıyla valide edilmiş, Dice katsayısı ve Hausdorff mesafesi gibi metrikler akademik eşiklerin üzerine çıkmıştır.
- **Fonksiyonel Prototip:** Segmentasyon, sağkalım ve radyogenomik tahmin modülleri birbiriyle entegre şekilde çalışır hale getirilmiş ve bir masaüstü/API prototipi oluşturulmuştur.
- **Laboratuvar Ortamı:** Tüm bileşenler simüle edilmiş MRI verileriyle laboratuvar ortamında (GPU kümesinde) test edilmiştir. Hazırlanan prototipin gerçek operasyonel ortamda (Hastane Bilgi Yönetim Sistemi - HBYS) pilot uygulaması henüz gerçekleştirilmediği için seviye 4'e geçiş planlama aşamasındadır.

---

### 📚 REFERANSLAR VE ETİK BEYAN
Proje; **KVKK (6698 Sayılı Kanun)** ve **WMA Helsinki Bildirgesi** etik ilkelerine tam uyumlu tasarlanmıştır. Veriler anonymize edilmiş olup projenin gelişim aşamalarında kullanılan tüm açık kaynaklı veri setleri (TCGA-GBM, BraTS) için atıf kuralları eksiksiz uygulanmıştır.

---
**NOT:** Bu rapor, metin içinde tekrara düşmeden teknik derinliği maksimize edecek şekilde tasarlanmıştır. Teslim öncesi projenize özel takım adı ve ID'leri ilgili alanlara eklemeyi unutmayınız.
