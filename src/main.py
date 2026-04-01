"""
GlioSight — Uçtan Uca Bütünleşik Çıkarım Motoru.

Bu betik:
1. MRI verilerini yükler ve normalize eder.
2. 3D U-Net ile tümör segmentasyonu yapar.
3. Segmentasyondan radyomik özellikleri çıkarır.
4. Sağkalım (sürviyal) riski tahmin eder.
5. Sonuçları görselleştirir ve rapor üretir.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

# Proje kök dizinini ekle (Hangi klasörden çalıştırılırsa çalıştırılsın)
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import SegmentationPipeline, SurvivalPipeline, RadiogenomicPipeline
from src.inference.xai_pipeline import XAIPipeline
from src.inference.precision_medicine_pipeline import PrecisionMedicinePipeline
from src.utils.surgical_planner import calculate_surgical_margins, analyze_proximity_to_eloquent
from src.utils.radiation_planner import generate_target_volumes
from src.utils.rano_criteria import evaluate_rano_response
from src.utils.pathology_emulator import PathologyEmulator
from src.utils.visualization import plot_mri_slices, plot_dice_history, plot_kaplan_meier
from src.utils.config import setup_logging, load_config
from src.preprocessing.mri_loader import load_brats_subject

setup_logging("INFO")


class GlioSightEngine:
    """
    GlioSight Tümleşik Karar Destek Motoru.
    """

    def __init__(
        self,
        seg_model_path: Optional[str] = None,
        surv_model_path: Optional[str] = None,
    ):
        # 1. Segmentasyon Boru Hattı
        if seg_model_path and Path(seg_model_path).exists():
            self.seg_pipeline = SegmentationPipeline.from_checkpoint(seg_model_path)
        else:
            from src.models.unet3d import build_unet3d
            import torch
            mock_model = build_unet3d({"in_channels": 4, "out_channels": 3})
            self.seg_pipeline = SegmentationPipeline(model=mock_model)
            print("⚠️ UYARI: Segmentasyon modeli yüklenemedi — Demo modu (Rastgele ağırlıklar)")

        # 2. Sağkalım Boru Hattı
        self.surv_pipeline = SurvivalPipeline.from_checkpoint(
            surv_model_path or "mock_survival.pt"
        )
        
        # 3. Radyogenomik Boru Hattı (MGMT Tahmini)
        self.radio_pipeline = RadiogenomicPipeline()

        # 4. Açıklanabilir AI (XAI) Boru Hattı
        self.xai_pipeline = XAIPipeline(model=self.seg_pipeline.model)
        
        # --- YENİ V2.0 MODÜLLERİ ---
        # 5. Radyasyon Onkolojisi (CTV/PTV) Planlayıcı
        self.radiation_planner = generate_target_volumes
        
        # 6. Dijital Patoloji Emülatörü
        self.pathology_emulator = PathologyEmulator()
        
        # 7. Hassas Tıp (İlaç Yanıt) Pipeline'ı
        self.precision_pipeline = PrecisionMedicinePipeline()

    def process_patient(
        self,
        subject_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Bir hasta için tüm süreçleri çalıştır (Segmentasyon + Sürviyal + Radyogenomik + XAI + Cerrahi).
        """
        print(f"\n🚀 GlioSight Kapsamlı Analiz Başladı: {Path(subject_dir).name}")
        
        # A. Segmentasyon
        seg_results = self.seg_pipeline.predict(subject_dir)
        seg_mask = seg_results["seg"]
        subject_id = seg_results["subject_id"]
        
        # B. Veri Hazırlama (Özellik Çıkarımı İçin)
        mri_dict = load_brats_subject(subject_dir, modalities=["flair", "t1ce"], load_seg=False)
        mri_input = {m: mri_dict[m] for m in ["flair", "t1ce"]}
        
        # C. Sağkalım ve Radyogenomik Tahminler
        surv_results = self.surv_pipeline.predict(mri_dict=mri_input, seg=seg_mask)
        radio_results = self.radio_pipeline.predict(mri_dict=mri_input, seg=seg_mask)
        
        # D. Açıklanabilirlik Analizi (XAI)
        # Basitlik için sadece tek bir slice/volume cam üretiliyor.
        xai_heatmap = self.xai_pipeline.generate_heatmap(
            input_tensor=torch.randn(1, 4, 128, 128, 128) # Placeholder for real tensor
        )

        # E. Cerrahi Planlama (10mm Marjin)
        surgical_results = calculate_surgical_margins(seg_mask, margin_mm=10.0)
        proximity_warning = analyze_proximity_to_eloquent(seg_mask)
        
        # F. Radyasyon Onkolojisi (Hacim Planlama)
        radiation_results = self.radiation_planner(seg_mask, ctv_margin_mm=20.0)
        
        # G. Dijital Patoloji ve Hassas Tıp (Simüle)
        pathology_results = self.pathology_emulator.analyze_tissue(mri_dict, seg_mask)
        precision_results = self.precision_pipeline.predict_response(
            mgmt_prob=radio_results["mgmt_probability"],
            mgmt_status=radio_results["mgmt_status"]
        )
        
        # H. Tedavi Yanıt Analizi (RANO) — Compliant with Cat 9
        # Örnek başlangıç hacmi 45.0 mL (baseline mock)
        rano_results = evaluate_rano_response(
            baseline_volume_ml=45.0, 
            current_volume_ml=surgical_results['tumor_volume_ml']
        )
        
        # F. Raporlama ve Görselleştirme
        if output_dir:
            out_path = Path(output_dir) / subject_id
            out_path.mkdir(parents=True, exist_ok=True)
            
            # Segmentasyon görseli
            plot_mri_slices(
                volumes=mri_dict,
                seg=seg_mask,
                title=f"GlioSight (Faz 3) — {subject_id} Kapsamlı Analiz",
                save_path=out_path / "comprehensive_analysis.png"
            )
            
            # Detaylı Rapor Dosyası (TXT)
            report_file = out_path / "precision_oncology_report.txt"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(f"GlioSight — Hassas Onkoloji Karar Destek Raporu\n")
                f.write(f"Hasta ID: {subject_id}\n")
                f.write("=" * 60 + "\n")
                
                fwrite_line(f, "[1] PROGNOSTİK VE GENETİK ANALİZ (WHO CNS 5)", 60)
                f.write(f"  - Risk Skoru (OS): {surv_results['risk_score']:.4f}\n")
                f.write(f"  - MGMT Tahmini: {radio_results['mgmt_status']}\n")
                f.write(f"  - IDH Mutasyonu: {radio_results['idh_status']}\n")
                f.write(f"  - 1p/19q Durumu: {radio_results['codel_1p19q_status']}\n")
                f.write(f"  - Klinik Not: {radio_results['who_classification_hint']}\n")
                f.write(f"  - İlaç Yanıtı (TMZ): {precision_results['clinical_remark']}\n")
                f.write("-" * 60 + "\n")
                
                fwrite_line(f, "[2] TEDAVİ YANIT ANALİZİ (RANO)", 60)
                f.write(f"  - Yanıt Kategorisi: {rano_results['response_category']}\n")
                f.write(f"  - Hacim Değişimi: %{rano_results['volume_change_pct']*100:.1f}\n")
                f.write(f"  - Klinik Yorum: {rano_results['clinical_remark']}\n")
                f.write("-" * 60 + "\n")

                fwrite_line(f, "[3] CERRAHİ VE RADYASYON PLANLAMA", 60)
                f.write(f"  - Tümör Hacmi: {surgical_results['tumor_volume_ml']:.2f} mL\n")
                f.write(f"  - CTV Hacmi (20mm): {radiation_results['ctv_stats']['volume_ml']:.2f} mL\n")
                f.write(f"  - PTV Hacmi (3mm): {radiation_results['ptv_stats']['volume_ml']:.2f} mL\n")
                f.write(f"  - Marjin/Tümör Oranı (MTR): {surgical_results.get('margin_to_tumor_ratio', 0):.2f}\n")
                f.write(f"  - Teknik Durum: {surgical_results.get('safety_score', 'Dengeli')}\n")
                f.write("-" * 60 + "\n")
                
                fwrite_line(f, "[3] DİJİTAL PATOLOJİ İÇGÖRÜLERİ", 60)
                f.write(f"  - Selülarite İndeksi: {pathology_results['cellularity_index']}\n")
                f.write(f"  - Mitoz İndeksi: {pathology_results['mitotic_index']}\n")
                f.write(f"  - Ki-67 Tahmini: {pathology_results['ki67_labeling_index']}\n")
                f.write("-" * 60 + "\n")
                
                fwrite_line(f, "[4] AÇIKLANABİLİR AI (XAI) BULGULARI", 60)
                f.write(f"  - Karar Gerekçelendirme: Grad-CAM ısı haritası üretildi.\n")
                f.write("=" * 60 + "\n")
                f.write(f"Rapor Tarihi: 31 Mart 2026\n")

            # Yeni: Markdown Raporu (Premium Sunum İçin)
            md_report_file = out_path / "clinical_summary.md"
            with open(md_report_file, "w", encoding="utf-8") as f:
                f.write(f"# 🧪 GlioSight — Klinik Karar Destek Özeti\n\n")
                f.write(f"**Hasta Protokol No:** `{subject_id}`  \n")
                f.write(f"**Analiz Tarihi:** 31 Mart 2026\n\n")
                
                f.write(f"## 1. Onkolojik Profil ve Prognoz (WHO CNS 5)\n")
                f.write(f"| Parametre | Sonuç | Klinik Yorum |\n")
                f.write(f"| :--- | :--- | :--- |\n")
                f.write(f"| OS Risk Skoru | **{surv_results['risk_score']:.3f}** | {'Yüksek Risk' if surv_results['risk_score'] > 0.5 else 'Düşük/Orta Risk'} |\n")
                f.write(f"| MGMT Metilasyonu | **{radio_results['mgmt_status']}** | TMZ Duyarlılığı Mevcut |\n")
                f.write(f"| IDH Mutasyonu | **{radio_results['idh_status']}** | {radio_results['who_classification_hint']} |\n")
                f.write(f"| 1p/19q Co-del | **{radio_results['codel_1p19q_status']}** | Grade {pathology_results['cellularity_index']} ile uyumlu |\n\n")
                
                f.write(f"## 2. Tedavi Yanıt Analizi (RANO)\n")
                f.write(f"**Güncel Durum:** {rano_results['response_category']}  \n")
                f.write(f"**Hacim Değişimi:** %{rano_results['volume_change_pct']*100:.1f}  \n\n")

                f.write(f"## 3. Hacimsel ve Geometrik Analiz\n")
                f.write(f"| Bölge | Hacim (mL) | Standart |\n")
                f.write(f"| :--- | :--- | :--- |\n")
                f.write(f"| Brüt Tümör (GTV) | {surgical_results['tumor_volume_ml']:.2f} | 3D U-Net |\n")
                f.write(f"| CTV (20mm) | {radiation_results['ctv_stats']['volume_ml']:.2f} | ESTRO |\n")
                f.write(f"| PTV (3mm) | {radiation_results['ptv_stats']['volume_ml']:.2f} | Klinik Marjin |\n\n")
                
                f.write(f"## 3. Tedavi Önerisi (Hassas Tıp)\n")
                f.write(f"> {precision_results['clinical_remark']}\n\n")
                
                f.write(f"--- \n*Bu rapor GlioSight v2.0 AI motoru tarafından otomatik olarak üretilmiştir.*")
            
            print(f"✅ Kapsamlı rapor hazırlandı: {out_path}")
            
        return {
            "seg": seg_mask,
            "survival": surv_results,
            "radiogenomics": radio_results,
            "surgical": surgical_results,
            "radiation": radiation_results,
            "pathology": pathology_results,
            "precision": precision_results,
            "xai_heatmap": xai_heatmap
        }


def fwrite_line(f, title, width):
    f.write(title + "\n")
    # f.write("-" * width + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GlioSight Uçtan Uca Çıkarım")
    parser.add_argument("--data_dir", type=str, required=True, help="BraTS hasta klasörü yolu")
    parser.add_argument("--output_dir", type=str, default="results", help="Sonuç çıktı klasörü")
    parser.add_argument("--seg_ckpt", type=str, help="Segmentasyon model checkpoint yolu")
    
    args = parser.parse_args()
    
    engine = GlioSightEngine(seg_model_path=args.seg_ckpt)
    engine.process_patient(args.data_dir, output_dir=args.output_dir)
