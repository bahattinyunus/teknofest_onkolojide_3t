"""
Eğitim Döngüsü — 3D U-Net BraTS Segmentasyon Eğitimi.

Özellikler:
    - Mixed precision eğitimi (torch.cuda.amp)
    - Otomatik model kaydı (en iyi validation Dice)
    - TensorBoard / MLflow loglama
    - Early stopping
    - LR scheduler (ReduceLROnPlateau / CosineAnnealing)
    - Deep supervision desteği
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..training.losses import DiceFocalLoss, build_loss
from ..training.metrics import MetricTracker, compute_segmentation_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    3D U-Net Segmentasyon Eğitici.

    Args:
        model          : UNet3D modeli
        train_loader   : Eğitim DataLoader
        val_loader     : Doğrulama DataLoader
        config         : Eğitim konfigürasyonu
        device         : Eğitim cihazı (cuda / cpu / mps)
        output_dir     : Model ve log dosyaları çıktı dizini
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: Optional[torch.device] = None,
        output_dir: str = "experiments/results",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cihaz seçimi
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Eğitim cihazı: {self.device}")
        self.model = self.model.to(self.device)

        # Loss
        self.criterion = build_loss(config.get("loss", {"type": "dice_focal"}))

        # Optimizer
        lr = config.get("lr", 1e-4)
        weight_decay = config.get("weight_decay", 1e-5)
        self.optimizer = AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Scheduler
        scheduler_type = config.get("scheduler", "cosine")
        epochs = config.get("epochs", 300)
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", patience=20, factor=0.5
            )

        # Mixed precision
        self.use_amp = config.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Early stopping
        self.patience = config.get("patience", 50)
        self._best_dice = -1.0
        self._patience_counter = 0
        self._best_model_path: Optional[Path] = None

        # Metrik takibi
        self.train_tracker = MetricTracker()
        self.val_tracker = MetricTracker()
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "val_dice_wt": [], "val_dice_tc": [], "val_dice_et": [],
        }

        # TensorBoard
        self._setup_logging()

    def _setup_logging(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = self.output_dir / "tensorboard"
            self.writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard: {log_dir}")
        except ImportError:
            self.writer = None

    def _to_device(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch'i cihaza taşı ve one-hot maske oluştur."""
        images = batch["image"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True)

        # One-hot kodlama (WT, TC, ET)
        num_classes = self.config.get("out_channels", 3)
        target = self._seg_to_onehot(seg, num_classes)
        return images, target

    @staticmethod
    def _seg_to_onehot(
        seg: torch.Tensor,
        num_classes: int = 3,
    ) -> torch.Tensor:
        """
        BraTS segmentasyon maskesini one-hot binary'e dönüştür.

        Sınıflar: [WT, TC, ET]
        """
        B = seg.shape[0]
        H, W, D = seg.shape[1:]
        onehot = torch.zeros(B, num_classes, H, W, D, device=seg.device)

        # WT = label 1,2,4
        onehot[:, 0] = (seg == 1) | (seg == 2) | (seg == 4)
        # TC = label 1,4
        onehot[:, 1] = (seg == 1) | (seg == 4)
        # ET = label 4
        if num_classes > 2:
            onehot[:, 2] = (seg == 4)

        return onehot.float()

    def _train_epoch(self, epoch: int) -> float:
        """Tek eğitim epoch'u. Ortalama loss döner."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:03d} [Train]",
            leave=False,
        )

        for batch in pbar:
            images, target = self._to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                if isinstance(outputs, (list, tuple)):
                    # Deep supervision
                    loss = self.criterion(list(outputs), target)
                    main_out = outputs[0]
                else:
                    loss = self.criterion(outputs, target)
                    main_out = outputs

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=12.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / n_batches

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Tuple[float, Dict]:
        """Tek doğrulama epoch'u. (val_loss, metrics) çifti döner."""
        self.model.eval()
        total_loss = 0.0
        dice_scores = {"WT": [], "TC": [], "ET": []}

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch:03d} [Val]",
            leave=False,
        )

        for batch in pbar:
            images, target = self._to_device(batch)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                if isinstance(outputs, (list, tuple)):
                    main_out = outputs[0]
                    loss = self.criterion(list(outputs), target)
                else:
                    main_out = outputs
                    loss = self.criterion(main_out, target)

            total_loss += loss.item()

            # Numpy'e çevir ve metrik hesapla
            probs = torch.sigmoid(main_out).cpu().numpy()
            target_np = target.cpu().numpy().astype(int)

            for b in range(probs.shape[0]):
                for c, region in enumerate(["WT", "TC", "ET"]):
                    pred_bin = (probs[b, c] > 0.5).astype(int)
                    gt_bin = target_np[b, c]
                    from ..training.metrics import dice_coefficient
                    d = dice_coefficient(pred_bin, gt_bin)
                    dice_scores[region].append(d)

        metrics = {
            region: float(np.mean(scores))
            for region, scores in dice_scores.items()
        }
        mean_dice = np.mean(list(metrics.values()))
        metrics["mean_dice"] = float(mean_dice)

        return total_loss / len(self.val_loader), metrics

    def _save_checkpoint(self, epoch: int, val_dice: float) -> Path:
        """Model checkpoint kaydet."""
        ckpt_path = self.output_dir / f"best_model_dice{val_dice:.4f}_ep{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_dice": val_dice,
                "config": self.config,
            },
            ckpt_path,
        )
        logger.info(f"✅ Checkpoint kaydedildi: {ckpt_path.name}")
        return ckpt_path

    def fit(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Modeli eğit.

        Args:
            epochs: Eğitim epoch sayısı (None → config'den alır)

        Returns:
            Eğitim geçmişi dict'i
        """
        max_epochs = epochs or self.config.get("epochs", 300)
        logger.info(f"Eğitim başlıyor | {max_epochs} epoch | device: {self.device}")

        for epoch in range(1, max_epochs + 1):
            t_start = time.time()

            # Eğitim
            train_loss = self._train_epoch(epoch)
            # Doğrulama
            val_loss, val_metrics = self._val_epoch(epoch)

            # Scheduler adımı
            mean_dice = val_metrics["mean_dice"]
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(mean_dice)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start

            # Loglama
            logger.info(
                f"Ep {epoch:03d}/{max_epochs} | "
                f"TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
                f"WT: {val_metrics['WT']:.4f} TC: {val_metrics['TC']:.4f} "
                f"ET: {val_metrics['ET']:.4f} | "
                f"MeanDice: {mean_dice:.4f} | LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Geçmiş kayıt
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice_wt"].append(val_metrics["WT"])
            self.history["val_dice_tc"].append(val_metrics["TC"])
            self.history["val_dice_et"].append(val_metrics["ET"])

            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Dice/WT", val_metrics["WT"], epoch)
                self.writer.add_scalar("Dice/TC", val_metrics["TC"], epoch)
                self.writer.add_scalar("Dice/ET", val_metrics["ET"], epoch)
                self.writer.add_scalar("Dice/mean", mean_dice, epoch)
                self.writer.add_scalar("LR", current_lr, epoch)

            # En iyi model kaydı
            if mean_dice > self._best_dice:
                self._best_dice = mean_dice
                self._patience_counter = 0
                if self._best_model_path and self._best_model_path.exists():
                    self._best_model_path.unlink()
                self._best_model_path = self._save_checkpoint(epoch, mean_dice)
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping: {self.patience} epoch'tur iyileşme yok | "
                        f"Best Dice: {self._best_dice:.4f}"
                    )
                    break

        if self.writer:
            self.writer.close()

        logger.info(f"Eğitim tamamlandı | En iyi Mean Dice: {self._best_dice:.4f}")
        return self.history
