# TEKNOFEST 2026 ONKOLOJİDE 3T YARIŞMASI
## PROJE ÖN DEĞERLENDİRME RAPORU (ÖDR) — OPTİMİZE EDİLMİŞ VERSİYON

---

### 1. BİYOTEKNOLOJİ ALANI
**GlioSight**, beyin kanserinin en agresif türü olan **Glioblastoma (GBM)** için geliştirilmiş bütünleşik bir dijital onkoloji sistemidir. Proje; **Radyoloji ve Görüntüleme Teknolojileri** (3B MRI analizi), **Cerrahi Onkoloji** (Marjin planlama), **Yapay Zekâ Destekli İlaç Geliştirme** (MGMT tahmini) ve **Tıbbi Onkoloji** (Sağkalım analizi) kategorilerini tek bir mimaride birleştirerek şartname standartlarını tam karşılar.

### 2. PROJE ÖZETİ
**Genel Hedef:** Glioblastoma hastalarında tanı, cerrahi planlama ve tedavi yanıtı takibini yapay zekâ ile standardize ederek hekim kararlarını veriye dayalı hale getirmektir.
**Hipotez:** MRI hacimlerinden çıkarılan radyomik öznitelikler, tümörün genetik profilini (MGMT metilasyonu) ve sağkalım süresini matematiksel olarak temsil eden dijital biyobelirteçlerdir.
**Yaklaşım:** Proje; **3D Residual U-Net** ile otomatik segmentasyon yaparken, Cox PH modelleri ile sağkalım tahmini ve Grad-CAM ile "Açıklanabilir AI" (XAI) arayüzü sunar.

![Görsel 1: GlioSight Bütünleşik Analiz Paneli ve Karar Destek Arayüzü](file:///g:/Di%C4%9Fer%20bilgisayarlar/Diz%C3%BCst%C3%BC%20Bilgisayar%C4%B1m/github%20repolar%C4%B1m/teknofest_onkolojide_3t/assets/gliosight_dashboard.png)

**Çıktılar:** (1) Otomatik 3B Segmentasyon, (2) Sağkalım ve Risk Analiz Raporu, (3) MGMT Metilasyon Tahmini ve (4) PACS uyumlu REST API mimarisi.

### 3. SORUN TANIMI
**Hedef Sorun:** Beyin kanseri yönetimindeki operasyonel gecikmeler ve prognostik belirsizliklerdir.
- **Segmentasyon Hatası:** Manuel analizler zaman alıcıdır ve radyologlar arası %20’ye varan değişkenliğe açıktır.
- **Tanısal Boşluk:** Mevcut görüntüleme yazılımları, hastanın genetik mutasyonu (MGMT) veya sağkalım beklentisi hakkında sayısal bir veri sunmamaktadır.
- **Cerrahi Risk:** Tümör sınırlarının (infiltrasyon) belirsizliği, yetersiz rezeksiyon veya sağlıklı doku hasarı riskini artırarak nüks oranını yükseltmektedir.

### 4. ÇÖZÜM
GlioSight, sorunları üç aşamalı bir yaklaşımla çözer:
- **Teknoloji:** Raw MRI verileri monai framework'ünde eğitilmiş 3B modellerle 1mm³ hassasiyetle işlenir.
- **Özgünlük:** Proje, sadece segmentasyon yapmakla kalmaz; segmentasyon sonuçlarından otomatik "Cerrahi Güvenlik Marjini" haritası çıkararak cerraha somut plan sunar.
- **Klinik Katkı:** Biyopsi sonucu beklenirken hastanın TMZ kemoterapisine duyarlılığını (MGMT tahmini) öngörerek tedavinin hızlandırılmasına katkı sağlar.

### 5. YENİLİKÇİ YÖNÜ VE ÖZGÜN DEĞERİ
**Teknolojik Yenilik (XAI):** GlioSight, kara kutu AI modellerinin aksine **Açıklanabilir AI (Grad-CAM)** kullanarak kararlarını anatomik ısı haritalarıyla gerekçelendirir.

![Görsel 2: XAI Isı Haritası (Grad-CAM) ve Cerrahi Güvenlik Marjini (10mm) Analizi](file:///g:/Di%C4%9Fer%20bilgisayarlar/Diz%C3%BCst%C3%BC%20Bilgisayar%C4%B1m/github%20repolar%C4%B1m/teknofest_onkolojide_3t/assets/gliosight_xai_surgical.png)

**Literatürden Farkı:** Klasik yöntemler tek modalite veya 2B kesitler üzerinde çalışırken, GlioSight **Çok Modlu (Multimodal) Hacimsel Füzyon** yaparak tümörün 3B geometrisini ve doku dokusunu derinlemesine analiz eder.
**Alana Katkı:** Tanı, genetik tahmin ve cerrahi planlamayı tek bir hibrit pipeline’da birleştiren yapısıyla "Hassas Onkoloji" standartlarını uçtan uca karşılamaktadır.

### 6. TEKNOLOJİ HAZIRLIK SEVİYESİ (THS)
**Konumlandırma:** Proje şu anda **THS 3 (Konsept Kanıtlanmış)** seviyesindedir.
**Dayanak:** 
- Algoritmalar, uluslararası **BraTS** veri setleri (500+ hasta) üzerinde valide edilerek akademik performans eşikleri (Dice Score, C-Index) aşılmıştır.
- Tüm bileşenler laboratuvar ortamında entegre şekilde çalışır hale getirilerek işlevsel bir API prototipi oluşturulmuştur.
- Klinik operasyonel ortamda (Hastane PACS ağı) canlı veriyle pilot uygulama aşamasına henüz geçilmediği için seviye 4 öncesi validasyon tamamlanmıştır.

---

### 📚 REFERANSLAR VE ETİK BEYAN
Proje; KVKK ve WMA Helsinki Bildirgesi ilkelerine tam uyumlu tasarlanmıştır. Veriler anonymize edilmiş ve BraTS/TCGA veri seti atıfları akademik formatta (IEEE TMI, ICCV) uygulanmıştır.

---
**NOT:** Bu rapor, sayfa sınırlarını korumak ve teknik otoriteyi sürdürmek için optimize edilmiştir. Takım adı ve ID bilgilerini ekleyerek sisteme yükleyebilirsiniz.
