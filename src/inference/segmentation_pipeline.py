"""
Segmentasyon Çıkarım Pipeline'ı — Eğitilmiş 3D U-Net ile Uçtan Uca Tahmin.

Özellikler:
    - Sliding window inference (patch-based, MONAI)
    - TTA (Test-Time Augmentation) — isteğe bağlı
    - NIfTI formatında çıkış kaydetme
    - Toplu hasta işleme
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def sliding_window_inference(
    model: nn.Module,
    image: torch.Tensor,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 2,
    overlap: float = 0.5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Sliding window çıkarımı (büyük hacimler için patch tabanlı tahmin).

    Args:
        model        : Eğitilmiş segmentasyon modeli
        image        : (1, C, H, W, D) veya (C, H, W, D) giriş tensörü
        roi_size     : Patch boyutu
        sw_batch_size: Aynı anda işlenen patch sayısı
        overlap      : Patch örtüşme oranı [0, 1)
        device       : Çıkarım cihazı

    Returns:
        (1, num_classes, H, W, D) olasılık haritası
    """
    try:
        from monai.inferers import sliding_window_inference as monai_swi
        return monai_swi(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
    except ImportError:
        logger.warning("MONAI yüklü değil — basit center crop kullanılıyor")
        return _simple_center_inference(model, image, roi_size, device)


def _simple_center_inference(
    model: nn.Module,
    image: torch.Tensor,
    roi_size: Tuple[int, int, int],
    device: Optional[torch.device],
) -> torch.Tensor:
    """MONAI olmadan basit merkez kırpma tabanlı çıkarım."""
    if image.dim() == 4:
        image = image.unsqueeze(0)

    _, C, H, W, D = image.shape
    rH, rW, rD = roi_size

    x = max(0, (H - rH) // 2)
    y = max(0, (W - rW) // 2)
    z = max(0, (D - rD) // 2)

    patch = image[:, :, x:x+rH, y:y+rW, z:z+rD]
    if device:
        patch = patch.to(device)

    with torch.no_grad():
        out = model(patch)
        if isinstance(out, (list, tuple)):
            out = out[0]

    # Pad back to original size
    full = torch.zeros(1, out.shape[1], H, W, D, device=out.device)
    full[:, :, x:x+rH, y:y+rW, z:z+rD] = out
    return full


def logits_to_segmentation(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Logitlerden BraTS formatında segmentasyon maskesi üret.

    Sınıflar: [WT, TC, ET] → BraTS etiketleri: {0, 1, 2, 4}

    Args:
        logits    : (1, 3, H, W, D) veya (3, H, W, D)
        threshold : Binary eşik

    Returns:
        (H, W, D) segmentasyon maskesi (0, 1, 2, 4 etiketli)
    """
    if logits.dim() == 5:
        logits = logits.squeeze(0)

    probs = torch.sigmoid(logits).cpu().numpy()  # (3, H, W, D)

    H, W, D = probs.shape[1:]
    seg = np.zeros((H, W, D), dtype=np.int32)

    wt = probs[0] > threshold
    tc = probs[1] > threshold
    et = probs[2] > threshold

    # ED (edema) = WT - TC → label 2
    seg[wt & ~tc] = 2
    # NCR/NET = TC - ET → label 1
    seg[tc & ~et] = 1
    # ET → label 4
    seg[et] = 4

    return seg


class SegmentationPipeline:
    """
    Uçtan Uca Segmentasyon Pipeline'ı.

    Kullanım:
        pipeline = SegmentationPipeline.from_checkpoint("best_model.pt")
        result = pipeline.predict(subject_dir="BraTS2021_00001/")
        pipeline.save_prediction(result["seg"], affine, "output.nii.gz")
    """

    def __init__(
        self,
        model: nn.Module,
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: float = 0.5,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
        use_tta: bool = False,
    ):
        self.model = model
        self.roi_size = roi_size
        self.overlap = overlap
        self.threshold = threshold
        self.use_tta = use_tta
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        **kwargs,
    ) -> "SegmentationPipeline":
        """
        Checkpoint dosyasından pipeline oluştur.

        Args:
            checkpoint_path : .pt dosya yolu
        """
        from ..models.unet3d import build_unet3d

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config = ckpt.get("config", {})
        model = build_unet3d(config.get("model", {}))
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Model yüklendi: {Path(checkpoint_path).name} | "
            f"Val Dice: {ckpt.get('val_dice', 'N/A')}"
        )
        return cls(model=model, **kwargs)

    def predict(
        self,
        subject_dir: Union[str, Path],
        modalities: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Hasta dizininden segmentasyon tahmini yap.

        Args:
            subject_dir : BraTS hasta dizini
            modalities  : Modalite listesi (default: T1, T1ce, T2, FLAIR)

        Returns:
            {"seg": (H,W,D), "probs": (3,H,W,D), "subject_id": str}
        """
        from ..preprocessing.mri_loader import load_brats_subject
        from ..preprocessing.normalization import normalize_multimodal

        if modalities is None:
            modalities = ["t1", "t1ce", "t2", "flair"]

        subject = load_brats_subject(subject_dir, modalities=modalities, load_seg=False)
        normalized = normalize_multimodal(
            {m: subject[m] for m in modalities},
            method="zscore",
        )

        # Stack → (C, H, W, D)
        stacked = np.stack([normalized[m] for m in modalities], axis=0)
        image_tensor = torch.from_numpy(stacked).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = sliding_window_inference(
                self.model, image_tensor,
                roi_size=self.roi_size,
                overlap=self.overlap,
                device=self.device,
            )
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

        seg = logits_to_segmentation(logits, threshold=self.threshold)
        probs = torch.sigmoid(logits.squeeze(0)).cpu().numpy()

        return {
            "seg": seg,
            "probs": probs,
            "subject_id": subject["subject_id"],
        }

    @staticmethod
    def save_prediction(
        seg: np.ndarray,
        affine: np.ndarray,
        output_path: Union[str, Path],
    ) -> None:
        """Tahmin maskesini NIfTI olarak kaydet."""
        import nibabel as nib
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img = nib.Nifti1Image(seg.astype(np.int16), affine)
        nib.save(img, str(output_path))
        logger.info(f"Segmentasyon kaydedildi: {output_path}")

    def batch_predict(
        self,
        subject_dirs: List[Union[str, Path]],
        output_dir: Union[str, Path],
    ) -> List[Dict]:
        """
        Toplu hasta tahmini.

        Args:
            subject_dirs : Hasta dizinleri listesi
            output_dir   : Çıkış dizini

        Returns:
            Her hasta için sonuç dict listesi
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for i, sdir in enumerate(subject_dirs):
            logger.info(f"İşleniyor [{i+1}/{len(subject_dirs)}]: {Path(sdir).name}")
            try:
                result = self.predict(sdir)
                results.append(result)
                logger.info(
                    f"  ✅ {result['subject_id']} | "
                    f"WT voxels: {(result['seg'] > 0).sum()}"
                )
            except Exception as e:
                logger.error(f"  ❌ Hata ({Path(sdir).name}): {e}")

        return results
