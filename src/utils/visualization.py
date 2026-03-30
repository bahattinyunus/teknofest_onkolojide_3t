"""
Görselleştirme Modülü — MRI ve Segmentasyon Görselleştirme.

Fonksiyonlar:
    - plot_mri_slices       : Çok modaliteli MRI eksensel kesit gösterimi
    - plot_segmentation_overlay : Tümör maskesi overlay (renkli)
    - plot_dice_history     : Eğitim Dice geçmişi grafiği
    - plot_kaplan_meier     : Sürviyal eğrisi
    - plot_shap_summary     : SHAP özellik önem grafiği
    - save_3d_gif           : 3B slice animasyonu GIF olarak kaydetme
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# BraTS renk haritası
BRATS_COLORS = {
    0: (0, 0, 0, 0),           # Arka plan (şeffaf)
    1: (1.0, 0.0, 0.0, 0.6),  # NCR/NET — Kırmızı
    2: (0.0, 0.8, 0.0, 0.6),  # ED (Edema) — Yeşil
    4: (1.0, 0.8, 0.0, 0.8),  # ET — Sarı
}


def plot_mri_slices(
    volumes: Dict[str, np.ndarray],
    slice_idx: Optional[int] = None,
    seg: Optional[np.ndarray] = None,
    cmap: str = "gray",
    title: str = "BraTS — Çok Modaliteli MRI",
    figsize: Tuple[int, int] = (18, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Çok modaliteli MRI kesitlerini yan yana göster.

    Args:
        volumes   : {"t1": arr, "t1ce": arr, "t2": arr, "flair": arr}
        slice_idx : Aksiyel kesit indeksi (None → merkez)
        seg       : İsteğe bağlı segmentasyon overlay
        cmap      : Renk haritası
        title     : Grafik başlığı
        figsize   : Figür boyutu
        save_path : Kayıt yolu (None → göster)

    Returns:
        matplotlib Figure
    """
    modalities = list(volumes.keys())
    n_cols = len(modalities) + (1 if seg is not None else 0)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", color="white")
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    for i, modality in enumerate(modalities):
        vol = volumes[modality]
        si = slice_idx if slice_idx is not None else vol.shape[2] // 2
        axes[i].imshow(vol[:, :, si].T, cmap=cmap, origin="lower", aspect="equal")
        axes[i].set_title(
            modality.upper(), color="white", fontsize=11, pad=8
        )
        axes[i].axis("off")

    if seg is not None:
        si = slice_idx if slice_idx is not None else seg.shape[2] // 2
        ax_seg = axes[-1]
        ref_vol = list(volumes.values())[1]  # T1ce referans
        ax_seg.imshow(ref_vol[:, :, si].T, cmap="gray", origin="lower")

        # Renkli overlay
        seg_rgb = _seg_to_rgb(seg[:, :, si])
        ax_seg.imshow(seg_rgb.transpose(1, 0, 2), origin="lower", alpha=0.6)

        # Gösterge
        patches = [
            mpatches.Patch(color=BRATS_COLORS[1][:3], label="NCR/NET"),
            mpatches.Patch(color=BRATS_COLORS[2][:3], label="Edema"),
            mpatches.Patch(color=BRATS_COLORS[4][:3], label="Enhancing"),
        ]
        ax_seg.legend(
            handles=patches, loc="lower right", fontsize=8,
            framealpha=0.8, facecolor="#1a1a2e", labelcolor="white"
        )
        ax_seg.set_title("Segmentasyon", color="white", fontsize=11, pad=8)
        ax_seg.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        logger.info(f"MRI görselleştirme kaydedildi: {save_path}")
    else:
        plt.show()

    return fig


def _seg_to_rgb(seg_slice: np.ndarray) -> np.ndarray:
    """(H, W) segmentasyon maskesini (H, W, 4) RGBA'ya dönüştür."""
    H, W = seg_slice.shape
    rgb = np.zeros((H, W, 4), dtype=np.float32)
    for label, color in BRATS_COLORS.items():
        mask = seg_slice == label
        for c in range(4):
            rgb[mask, c] = color[c]
    return rgb


def plot_dice_history(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Eğitim geçmişi grafiği — Loss ve Dice skorları.

    Args:
        history   : Trainer.history dict'i
        save_path : Kayıt yolu
        figsize   : Figür boyutu
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#16213e")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#0f3460")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#e94560")

    # Loss grafiği
    if "train_loss" in history:
        ax1.plot(history["train_loss"], label="Train Loss", color="#e94560", lw=2)
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="Val Loss", color="#0f3460", lw=2,
                 linestyle="--", color2=None)
        ax1.plot(history["val_loss"], label="Val Loss", color="#a8edea", lw=2, ls="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Eğitim Kaybı")
    ax1.legend(facecolor="#16213e", labelcolor="white")
    ax1.grid(alpha=0.2, color="white")

    # Dice grafiği
    colors = {"WD": "#e94560", "TC": "#a8edea", "ET": "#fed6e3"}
    for region, color in [("WT", "#e94560"), ("TC", "#a8edea"), ("ET", "#fed6e3")]:
        key = f"val_dice_{region.lower()}"
        if key in history:
            ax2.plot(history[key], label=f"Dice {region}", color=color, lw=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_title("Doğrulama Dice Skorları")
    ax2.legend(facecolor="#16213e", labelcolor="white")
    ax2.grid(alpha=0.2, color="white")
    ax2.set_ylim(0, 1)

    fig.suptitle("GlioSight — Eğitim Geçmişi", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#16213e")
    else:
        plt.show()

    return fig


def plot_kaplan_meier(
    time: np.ndarray,
    event: np.ndarray,
    risk_groups: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    title: str = "Kaplan-Meier Sürviyal Eğrisi",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Kaplan-Meier sürviyal eğrisi çiz.

    Args:
        time        : Sürviyal süreleri (gün)
        event       : Olay göstergesi (1=ölüm, 0=sansör)
        risk_groups : Risk grubu etiketleri (0=düşük, 1=orta, 2=yüksek)
        labels      : Risk grubu adları
        title       : Grafik başlığı
        save_path   : Kayıt yolu
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        logger.error("lifelines yüklü değil — pip install lifelines")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#16213e")
    ax.set_facecolor("#0f3460")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    if risk_groups is None:
        kmf = KaplanMeierFitter()
        kmf.fit(time, event, label="Tüm Hastalar")
        kmf.plot_survival_function(ax=ax, color="#a8edea", linewidth=2, ci_alpha=0.15)
    else:
        group_labels = labels or [f"Risk Grubu {i}" for i in range(len(np.unique(risk_groups)))]
        palette = ["#a8edea", "#e94560", "#fed093"]
        for g, (gid, color) in enumerate(zip(np.unique(risk_groups), palette)):
            mask = risk_groups == gid
            kmf = KaplanMeierFitter()
            kmf.fit(time[mask], event[mask], label=group_labels[g])
            kmf.plot_survival_function(ax=ax, color=color, linewidth=2, ci_alpha=0.1)

    ax.set_xlabel("Süre (Gün)", color="white")
    ax.set_ylabel("Sürviyal Olasılığı", color="white")
    ax.set_title(title, color="white", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor="#16213e", labelcolor="white")
    ax.grid(alpha=0.2, color="white")

    for spine in ax.spines.values():
        spine.set_edgecolor("#e94560")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#16213e")
    else:
        plt.show()

    return fig
