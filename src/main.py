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

from src.inference import SegmentationPipeline, SurvivalPipeline
from src.inference.radiogenomic_pipeline import RadiogenomicPipeline
from src.utils.visualization import plot_mri_slices, plot_dice_history, plot_kaplan_meier
from src.utils.config import setup_logging, load_config
from src.preprocessing.mri_loader import load_brats_subject

setup_logging("INFO")


class GlioSightEngine:
    """
    GlioSight Tümleşik Motoru.
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
            # Mock / Demo modu
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

    def process_patient(
        self,
        subject_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Bir hasta için tüm süreçleri çalıştır.
        """
        print(f"\n🚀 GlioSight İşleme Başladı: {Path(subject_dir).name}")
        
        # A. Segmentasyon
        seg_results = self.seg_pipeline.predict(subject_dir)
        seg_mask = seg_results["seg"]
        subject_id = seg_results["subject_id"]
        
        # B. Sağkalım Tahmini
        # MRI verilerini tekrar yükle (Hacim analizleri için)
        mri_dict = load_brats_subject(subject_dir, modalities=["flair", "t1ce"], load_seg=False)
        mri_input = {m: mri_dict[m] for m in ["flair", "t1ce"]}
        
        surv_results = self.surv_pipeline.predict(
            mri_dict=mri_input,
            seg=seg_mask
        )
        
        # C. Radyogenomik Analiz (MGMT Metilasyon Tahmini)
        radio_results = self.radio_pipeline.predict(
            mri_dict=mri_input,
            seg=seg_mask
        )
        
        # D. Raporlama ve Görselleştirme
        if output_dir:
            out_path = Path(output_dir) / subject_id
            out_path.mkdir(parents=True, exist_ok=True)
            
            # Segmentasyon görseli
            plot_mri_slices(
                volumes=mri_dict,
                seg=seg_mask,
                title=f"GlioSight — {subject_id} Segmentasyon Analizi",
                save_path=out_path / "segmentation.png"
            )
            
            # Rapor dosyası
            report_file = out_path / "report.txt"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(f"GlioSight Analiz Raporu — {subject_id}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Risk Skoru (Survival): {surv_results['risk_score']:.4f}\n")
                f.write(f"Risk Durumu: {surv_results['risk_level']}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Radyogenomik Bulgular (MGMT): {radio_results['mgmt_status']}\n")
                f.write(f"Metilasyon Olasılığı: %{radio_results['mgmt_probability'] * 100:.2f}\n")
                f.write("-" * 50 + "\n")
                f.write("Hacimsel Analiz:\n")
                for k, v in surv_results["features"].items():
                    if "volume" in k:
                        f.write(f"  - {k}: {v:.2f} mL\n")
                f.write("=" * 50 + "\n")
            
            print(f"✅ Sonuçlar kaydedildi: {out_path}")
            
        return {
            "seg": seg_mask,
            "survival": surv_results,
            "radiogenomics": radio_results
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GlioSight Uçtan Uca Çıkarım")
    parser.add_argument("--data_dir", type=str, required=True, help="BraTS hasta klasörü yolu")
    parser.add_argument("--output_dir", type=str, default="results", help="Sonuç çıktı klasörü")
    parser.add_argument("--seg_ckpt", type=str, help="Segmentasyon model checkpoint yolu")
    
    args = parser.parse_args()
    
    engine = GlioSightEngine(seg_model_path=args.seg_ckpt)
    engine.process_patient(args.data_dir, output_dir=args.output_dir)
