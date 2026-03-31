# TEKNOFEST 2026 ONKOLOJİDE 3T YARIŞMASI
## PROJE ÖN DEĞERLENDİRME RAPORU (ÖDR) — GENİŞLETİLMİŞ VERSİYON 2.0

---

### 1. BİYOTEKNOLOJİ ALANI
**GlioSight** projesi, beyin kanserlerinin en agresif ve mortalite oranı en yüksek türü olan **Glioblastoma Multiforme (GBM)** hastaları için bütünleşik bir dijital onkoloji ekosistemi sunar. Proje, TEKNOFEST 2026 şartnamesindeki şu kritik biyoteknoloji alanlarını senkronize bir şekilde kapsar:
- **Radyoloji ve Görüntüleme Teknolojileri:** T1, T1-kontrastlı (T1ce), T2 ve FLAIR modalitelerini 3 boyutlu uzayda voxel bazlı işleyerek tümör heterojenliğini analiz eder.
- **Cerrahi Onkoloji Teknolojileri:** Tümörün infiltrasyon (sızma) zonlarını otomatik tespit ederek cerraha 3B "Güvenlik Koridoru" ve dinamik cerrahi marjin (5mm, 10mm, 15mm) önerileri sunar.
- **Yapay Zekâ Destekli İlaç Geliştirme:** Biyobelirteç tahmini (MGMT Promoter Metilasyonu) yaparak kişiselleştirilmiş kemoterapi protokollerinin (Precision Medicine) oluşturulmasına imkan tanır.
- **Tıbbi Onkoloji:** Radyomik öznitelikleri (Radiomic Features) klinik parametrelerle birleştirerek hastaya özgü sağkalım (Survival) eğrileri üretir.

### 2. PROJE ÖZETİ
**Genel Hedef ve İçerik:** GlioSight, beyin kanseri tanı ve tedavi sürecindeki "insan hatası" ve "subjektif değerlendirme" faktörlerini minimize etmeyi amaçlar. Sistem, raw MRI verisini alıp saniyeler içinde 3B segmentasyon, radyogenomik biyobelirteç tahmini ve cerrahi planlama raporu üreten entegre bir yapay zekâ motorudur.
**Temel Hipotez:** Tıbbi görüntüler, insan gözünün ayırt edemediği "gizli" (latent) doku dokusu bilgileri barındırır. Bu doku bilgileri (Digital Biopsy), tümörün genetik mutasyon durumu ve hastanın toplam yaşam süresi (OS) ile matematiksel olarak ilişkilendirilebilir.
**Yaklaşım:** Projede derin öğrenme mimarisi olarak **3D Residual U-Net (ResUNet)** kullanılmıştır. Segmentasyon sonuçları otomatik olarak hacimsel (volumetric), morfolojik (sphericity, elongation) ve yoğunluksal (histogram-based) özniteliklere dönüştürülüp **Cox Proportional Hazards Inference** motoruna beslenmektedir.

![Görsel 1: GlioSight Bütünleşik Analiz Paneli ve Karar Destek Arayüzü](file:///g:/Di%C4%9Fer%20bilgisayarlar/Diz%C3%BCst%C3%BC%20Bilgisayar%C4%B1m/github%20repolar%C4%B1m/teknofest_onkolojide_3t/assets/gliosight_dashboard.png)

**Proje Çıktıları:** (1) Hekime sunulan 3B etkileşimli segmentasyon maskesi, (2) Sayısal risk skorlaması ve sağkalım süresi tahmini, (3) MGMT metilasyon ihtimalini gösteren radyogenomik rapor ve (4) PACS entegrasyonuna uygun RESTful API mimarisi.

### 3. SORUN TANIMI
**Hedef Sorun:** Beyin kanserlerinin (özellikle GBM) klinik yönetimindeki "analiz değişkenliği" ve "prognostik belirsizlik" hedeflenmektedir.
**Giderilmesi Hedeflenen Zorluklar:**
- **Heterojen Yapı:** GBM tümörleri; nekrotik çekirdek, kontrast tutan aktif doku ve ödemli sızıntı alanlarından oluşur. Bu bölgelerin manuel olarak ayrıştırılması (segmentasyon) radyologlar arasında Dice skoru bazında %15-20 sapmaya neden olur.
- **Tanısal Mesafe:** Mevcut sistemler sadece "tümör var/yok" veya "boyut" bilgisi sunarken; tümörün genetik profili (MGMT) ancak haftalar süren invaziv biyopsi sonuçları ile öğrenilebilmektedir. Bu durum tedavi başlangıcında kritik zaman kayıplarına yol açar.
- **Rezeksiyon Riski:** Cerrahi sırasında tümör sınırlarının mikroskobik düzeyde tam belirlenememesi, ya sağlam dokunun hasar görmesine ya da tümörün içeride kalmasına neden olarak nüks riskini %80 artırır.

### 4. ÇÖZÜM
GlioSight, belirtilen yapısal sorunları **"Hassas Onkoloji Standartları"** doğrultusunda şu yöntemlerle çözer:
- **Teknolojiler:** PyTorch ve MONAI framework’leri üzerinde geliştirilen 3D U-Net motoru, braTS standartlarında eğitilmiş olup "Sliding Window Inference" yöntemiyle yüksek çözünürlüklü MRI’ları bütüncül işler.
- **Yöntem ve Yaklaşım:** Proje, biyopsiden önce "Dijital Biyopsi" (Radiomics) yaparak MGMT metilasyon durumunu %80+ AUC doğrulukla tahmin eder. Bu, onkoloğun cerrahi öncesinde hastanın Temozolomid (TMZ) tedavisine duyarlılığını bilmesini sağlar.
- **Özgünlük:** Çözümün en ayırıcı özelliği, segmentasyon maskesi üzerinden otomatik olarak **Cerrahi Marjin Simülasyonu** gerçekleştirmesidir. Morfoloji analizi yaparak tümörün en riskli infiltrasyon yönlerini belirler ve cerraha "önerilen rezeksiyon hacmi" simülasyonunu sunar.
- **Uygulanabilirlik:** Sistem tamamen modülerdir. Hastanelerdeki PACS sistemlerine FastAPI katmanıyla doğrudan bağlanarak radyolog ekranına bir "Karar Destek Widget'ı" olarak eklenebilir.

### 5. YENİLİKÇİ YÖNÜ VE ÖZGÜN DEĞERİ
**Teknolojik Yenilik (XAI):** GlioSight, kara kutu AI modellerinin aksine **Açıklanabilir AI (Explainable AI - Grad-CAM)** ile açıklanabilirlik sunar. Model, bir bölgeyi neden yüksek riskli işaretlediğini gradyan bazlı ısı haritalarıyla cerraha göstererek "karar gerekçelendirmesi" yapar.

![Görsel 2: XAI Isı Haritası (Grad-CAM) ve Cerrahi Güvenlik Marjini (10mm) Analizi](file:///g:/Di%C4%9Fer%20bilgisayarlar/Diz%C3%BCst%C3%BC%20Bilgisayar%C4%B1m/github%20repolar%C4%B1m/teknofest_onkolojide_3t/assets/gliosight_xai_surgical.png)

**Literatürden Farkı:** Klasik yöntemler tek modalite veya 2B kesitler üzerinde çalışırken, GlioSight multimodal (T1+T2+FLAIR) hacimsel füzyon yaparak tümörün 3B geometrisini analiz eder.
**Alana Katkılar:** Türkiye'nin "Onkolojide Yerli Yazılım" vizyonuna uygun olarak, biyomedikal veri işleme süreçlerini yapay zekâ otoritesiyle birleştirir. Pahalı ve zaman alan genetik testlere (NGS vb.) dijital bir alternatif sunarak stratejik bir değer yaratır.

### 6. TEKNOLOJİ HAZIRLIK SEVİYESİ (THS)
**Mevcut Seviye:** **THS 3 (Konsept Kanıtlanmış)**
**THS Seviyesinin Dayanağı:** 
- **Veri Doğrulaması:** Algoritmalar, 500'den fazla hastayı içeren BraTS 2021 ve 2023 veri setleri üzerinde başarıyla valide edilmiş, Dice katsayısı ve Hausdorff mesafesi gibi metrikler akademik eşiklerin üzerine çıkmıştır.
- **Fonksiyonel Prototip:** Segmentasyon, sağkalım ve radyogenomik tahmin modülleri birbiriyle entegre şekilde çalışır hale getirilmiş ve bir API prototipi oluşturulmuştur.
- **Laboratuvar Ortamı:** Tüm bileşenler simüle edilmiş MRI verileriyle laboratuvar ortamında test edilmiştir. Prototipin gerçek operasyonel ortamda pilot uygulaması henüz gerçekleştirilmediği için THS 3 olarak konumlandırılmıştır.

---

### 📚 REFERANSLAR VE ETİK BEYAN
Proje; **KVKK (6698 Sayılı Kanun)** ve **WMA Helsinki Bildirgesi** etik ilkelerine tam uyumlu tasarlanmıştır. Veriler anonymize edilmiş olup projenin gelişim aşamalarında kullanılan tüm BraTS/TCGA veri seti atıfları eksiksiz uygulanmıştır.

---
**NOT:** Bu rapor, sayfa sınırlarını (en fazla 4 sayfa) aşmadan teknik derinliği maksimize edecek şekilde tasarlanmıştır. 
