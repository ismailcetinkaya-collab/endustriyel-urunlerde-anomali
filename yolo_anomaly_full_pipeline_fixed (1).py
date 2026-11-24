#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11 + SKU bazlı Autoencoder anomali tespit hattı.

Notlar:
- Sentetik veri, gerçek üretim verisinin yerine geçmez. Gerçek sahada kullanılmadan
  önce gerçek kamera görüntüleriyle doğrulanmalıdır.
- Veri seti train/val/test olarak ayrılır; YOLO doğrulaması eğitim verisini kullanmaz.
- Her ROI için SKU, anomali skoru ve eşik ayrı değerlendirilir.
"""

from __future__ import annotations

import os
# Bazı ortamlarda OpenMP kitaplıklarının çakışmasını azaltır.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import glob
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import onnxruntime as ort
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# ---------------------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------------------
SEED = 42
IMAGE_SIZE = 416
ROI_SIZE = 160
AE_SIZE = 256
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministik mod performansı bir miktar düşürebilir; tekrar üretilebilirlik sağlar.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_images(folder: str | Path) -> list[str]:
    folder = str(folder)
    return sorted(
        p for p in glob.glob(os.path.join(folder, "*"))
        if Path(p).suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def safe_imwrite(path: str | Path, image: np.ndarray) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if image is None or image.size == 0:
        raise ValueError(f"Kaydedilecek görüntü boş: {path}")
    if not cv2.imwrite(path, image):
        raise IOError(f"Görüntü diske yazılamadı: {path}")


# =====================================================================
# BÖLÜM 1: SENTETİK VERİ SETİ ÜRETİMİ
# =====================================================================
class ConveyorYOLOFactoryEnv:
    def __init__(
        self,
        base_dir: str = "yolo_vision_workspace",
        num_skus: int = 15,
        seed: int = SEED,
    ):
        if num_skus < 1:
            raise ValueError("num_skus en az 1 olmalıdır.")
        self.base_dir = Path(base_dir)
        self.num_skus = num_skus
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sku_names = [f"urun_{i:02d}" for i in range(1, num_skus + 1)]
        # Her SKU için sabit bir renk kimliği: aynı sınıf örnekleri tanınabilir kalır.
        palette_rng = np.random.default_rng(seed + 1000)
        self.sku_colors = [
            palette_rng.integers(75, 220, size=3, dtype=np.int16)
            for _ in range(num_skus)
        ]

        self.yolo_dir = self.base_dir / "yolo_dataset"
        self.anomaly_train_dir = self.base_dir / "anomaly_dataset"
        self.pool_6000_dir = self.base_dir / "gelen_6000_goruntu"
        self.gt_csv_path = self.base_dir / "ground_truth_6000.csv"

    def _draw_product_patch(self, sku_idx: int, patch_size: int = ROI_SIZE) -> np.ndarray:
        """SKU geometrisini koruyarak renk, doku ve yüzey görünümünü çeşitlendirir."""
        rng = self.rng
        patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        # Sınıf rengi sabit tutulur; örnekler arasında yalnızca kontrollü varyasyon olur.
        base_color = np.clip(
            self.sku_colors[sku_idx] + rng.integers(-12, 13, size=3), 55, 240
        )
        body_color = tuple(int(c) for c in base_color)
        center = (patch_size // 2, patch_size // 2)
        shape_type = sku_idx % 3

        if shape_type == 0:
            margin = int(rng.integers(10, 18))
            cv2.rectangle(patch, (margin, margin), (patch_size - margin, patch_size - margin), body_color, -1)
        elif shape_type == 1:
            radius = int(rng.integers(patch_size // 2 - 17, patch_size // 2 - 8))
            cv2.circle(patch, center, radius, body_color, -1)
        else:
            points = []
            radius = int(rng.integers(patch_size // 2 - 16, patch_size // 2 - 7))
            rotation = float(rng.uniform(0, 60))
            for k in range(6):
                angle = np.deg2rad(k * 60 + rotation)
                points.append([
                    int(center[0] + radius * np.cos(angle)),
                    int(center[1] + radius * np.sin(angle)),
                ])
            cv2.fillPoly(patch, [np.asarray(points, dtype=np.int32)], body_color)

        holes = (sku_idx % 4) + 1
        hole_radius = int(rng.integers(5, 9))
        for h_i in range(holes):
            angle = np.deg2rad(h_i * (360 / holes) + 45 + rng.uniform(-5, 5))
            hx = int(center[0] + 32 * np.cos(angle))
            hy = int(center[1] + 32 * np.sin(angle))
            cv2.circle(patch, (hx, hy), hole_radius, (25, 25, 28), -1)
            cv2.circle(patch, (hx, hy), hole_radius + 2, (235, 235, 235), 1)

        # Hafif yüzey dokusu yalnızca ürünün bulunduğu alana uygulanır.
        texture = rng.normal(0, 3.0, size=patch.shape).astype(np.int16)
        mask = np.any(patch > 0, axis=2)
        noisy = patch.astype(np.int16)
        noisy[mask] += texture[mask]
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _inject_defect_on_patch(self, patch: np.ndarray) -> tuple[np.ndarray, str]:
        out = patch.copy()
        h, w = out.shape[:2]
        kind = random.choice(["CIZIK", "LEKE", "EKSIK_ISLEM"])

        if kind == "CIZIK":
            x1, y1 = random.randint(35, 75), random.randint(35, 75)
            x2 = min(w - 5, x1 + random.randint(25, 65))
            y2 = max(5, min(h - 5, y1 + random.randint(-35, 35)))
            cv2.line(out, (x1, y1), (x2, y2), (250, 250, 255), thickness=random.randint(1, 3), lineType=cv2.LINE_AA)
        elif kind == "LEKE":
            cx, cy = random.randint(45, 115), random.randint(45, 115)
            axes = (random.randint(8, 18), random.randint(5, 13))
            cv2.ellipse(out, (cx, cy), axes, random.randint(0, 180), 0, 360, (18, 18, 22), -1)
        else:
            # Eksik işlem: ürün yüzeyinde beklenen bir işaret/işlem bölgesini silikleştirir.
            cx, cy = w // 2 + 22, h // 2 + 22
            cv2.circle(out, (cx, cy), random.randint(9, 15), (0, 0, 0), -1)
        return out, kind

    def _place_on_conveyor(
        self, patch: np.ndarray, frame_size: int = IMAGE_SIZE
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        rng = self.rng
        base = np.full((frame_size, frame_size, 3), (42, 45, 50), dtype=np.int16)
        noise = rng.integers(-4, 5, size=base.shape, dtype=np.int16)
        conveyor = np.clip(base + noise, 0, 255).astype(np.uint8)

        ph, pw = patch.shape[:2]
        max_x = frame_size - pw - 20
        max_y = frame_size - ph - 20
        if max_x < 20 or max_y < 20:
            raise ValueError("Ürün boyutu konveyör görüntüsünden büyük.")
        x1 = random.randint(20, max_x)
        y1 = random.randint(20, max_y)
        x2, y2 = x1 + pw, y1 + ph

        mask = np.any(patch > 0, axis=2)
        roi = conveyor[y1:y2, x1:x2]
        roi[mask] = patch[mask]
        conveyor[y1:y2, x1:x2] = roi
        return conveyor, (x1, y1, x2, y2)

    @staticmethod
    def _write_yolo_sample(
        image_dir: Path, label_dir: Path, filename: str,
        image: np.ndarray, sku_idx: int, box: tuple[int, int, int, int],
    ) -> None:
        x1, y1, x2, y2 = box
        xc = ((x1 + x2) / 2) / IMAGE_SIZE
        yc = ((y1 + y2) / 2) / IMAGE_SIZE
        bw = (x2 - x1) / IMAGE_SIZE
        bh = (y2 - y1) / IMAGE_SIZE
        safe_imwrite(image_dir / f"{filename}.jpg", image)
        with open(label_dir / f"{filename}.txt", "w", encoding="utf-8") as f:
            f.write(f"{sku_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    def build(
        self,
        yolo_train_per_sku: int = 60,
        anomaly_train_per_sku: int = 80,
        anomaly_calibration_per_sku: int = 30,
        total_test_images: int = 6000,
        defect_ratio: float = 0.15,
    ) -> str:
        if yolo_train_per_sku < 3:
            raise ValueError("YOLO train/val/test ayrımı için SKU başına en az 3 görüntü gerekir.")
        if min(anomaly_train_per_sku, anomaly_calibration_per_sku) < 2:
            raise ValueError("Anomali eğitim ve kalibrasyon örnek sayıları en az 2 olmalıdır.")
        if total_test_images < 1:
            raise ValueError("total_test_images en az 1 olmalıdır.")
        if not 0 <= defect_ratio <= 1:
            raise ValueError("defect_ratio 0 ile 1 arasında olmalıdır.")

        set_reproducibility(self.seed)
        # Yalnızca bu çalıştırmanın ürettiği veri klasörleri temizlenir.
        for path in (self.yolo_dir, self.anomaly_train_dir, self.pool_6000_dir):
            if path.exists():
                shutil.rmtree(path)
        if self.gt_csv_path.exists():
            self.gt_csv_path.unlink()

        for split in ("train", "val", "test"):
            (self.yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Her SKU kendi içinde karıştırılarak train/val/test'e ayrılır.
        for sku_idx, sku in enumerate(self.sku_names):
            samples = []
            for n in range(yolo_train_per_sku):
                patch = self._draw_product_patch(sku_idx)
                scene, box = self._place_on_conveyor(patch)
                samples.append((scene, box))
            random.shuffle(samples)
            n_total = len(samples)
            n_train = max(1, int(n_total * 0.70))
            n_val = max(1, int(n_total * 0.20))
            if n_train + n_val >= n_total:
                n_train = max(1, n_total - 2)
                n_val = 1
            splits = (
                ("train", samples[:n_train]),
                ("val", samples[n_train:n_train + n_val]),
                ("test", samples[n_train + n_val:]),
            )
            for split, split_samples in splits:
                for local_i, (scene, box) in enumerate(split_samples):
                    fname = f"{sku}_{split}_{local_i:05d}"
                    self._write_yolo_sample(
                        self.yolo_dir / "images" / split,
                        self.yolo_dir / "labels" / split,
                        fname, scene, sku_idx, box,
                    )

            good_dir = self.anomaly_train_dir / sku / "train" / "good"
            calib_dir = self.anomaly_train_dir / sku / "calibration" / "good"
            good_dir.mkdir(parents=True, exist_ok=True)
            calib_dir.mkdir(parents=True, exist_ok=True)
            for i in range(anomaly_train_per_sku):
                safe_imwrite(good_dir / f"{sku}_{i:05d}.png", self._draw_product_patch(sku_idx))
            for i in range(anomaly_calibration_per_sku):
                safe_imwrite(calib_dir / f"{sku}_{i:05d}.png", self._draw_product_patch(sku_idx))

        yaml_data = {
            "path": str(self.yolo_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(self.sku_names)},
        }
        yaml_path = self.yolo_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, allow_unicode=True, sort_keys=False)

        self.pool_6000_dir.mkdir(parents=True, exist_ok=True)
        with open(self.gt_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Dosya_Adi", "Gercek_Urun", "Gercek_Durum", "Kusur_Tipi"])
            for i in range(1, total_test_images + 1):
                sku_idx = random.randrange(self.num_skus)
                sku = self.sku_names[sku_idx]
                patch = self._draw_product_patch(sku_idx)
                if random.random() < defect_ratio:
                    patch, defect_type = self._inject_defect_on_patch(patch)
                    status = "NOK"
                else:
                    defect_type, status = "YOK", "OK"
                scene, _ = self._place_on_conveyor(patch)
                fname = f"BANT_{i:05d}.jpg"
                safe_imwrite(self.pool_6000_dir / fname, scene)
                writer.writerow([fname, sku, status, defect_type])

        print(
            f"[ORTAM HAZIR] {self.num_skus} SKU | "
            f"{total_test_images} test görüntüsü | YOLO train/val/test ayrıldı."
        )
        return str(yaml_path)


# =====================================================================
# BÖLÜM 2: DATASET VE AUTOENCODER
# =====================================================================
class ROIAnomalyDataset(Dataset):
    def __init__(self, image_dir: str | Path):
        self.paths = list_images(image_dir)
        if not self.paths:
            raise ValueError(f"Anomali veri klasörü boş: {image_dir}")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((AE_SIZE, AE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = cv2.imread(self.paths[idx], cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Görüntü okunamadı: {self.paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image)


class SurfaceConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _batch_anomaly_scores(model: nn.Module, loader: DataLoader, device: str) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for images in loader:
            images = images.to(device, non_blocking=True)
            reconstructed = model(images)
            error_map = torch.mean(torch.abs(images - reconstructed), dim=1)
            batch_scores = torch.quantile(error_map.flatten(1), 0.995, dim=1)
            scores.extend(float(v) for v in batch_scores.cpu().tolist())
    return scores


def train_yolo_and_anomalies(
    yaml_path: str,
    anomaly_root: str,
    models_dir: str,
    yolo_epochs: int = 30,
    ae_epochs: int = 40,
    batch_size: int = 16,
    calibration_quantile: float = 99.5,
) -> str:
    if yolo_epochs < 1 or ae_epochs < 1:
        raise ValueError("Epoch sayıları en az 1 olmalıdır.")
    if not 90 <= calibration_quantile <= 100:
        raise ValueError("calibration_quantile 90 ile 100 arasında olmalıdır.")

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_device: int | str = 0 if torch.cuda.is_available() else "cpu"

    print("\n--- AŞAMA 1: YOLO11 eğitimi ---")
    yolo_run = models_path / "yolo_detector"
    if yolo_run.exists():
        shutil.rmtree(yolo_run)
    yolo = YOLO("yolo11n.pt")
    yolo.train(
        data=yaml_path,
        epochs=yolo_epochs,
        imgsz=IMAGE_SIZE,
        batch=batch_size,
        device=yolo_device,
        project=str(models_path),
        name="yolo_detector",
        exist_ok=True,
        patience=10,
        workers=0,
        seed=SEED,
        deterministic=True,
        verbose=False,
    )
    best_weights = yolo_run / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"YOLO best.pt bulunamadı: {best_weights}")

    print("\n--- AŞAMA 2: SKU bazlı Autoencoder + ONNX ---")
    sku_names = sorted(
        d.name for d in Path(anomaly_root).iterdir()
        if d.is_dir() and (d / "train" / "good").is_dir()
    )
    if not sku_names:
        raise ValueError(f"SKU eğitim klasörü bulunamadı: {anomaly_root}")

    registry: dict[str, Any] = {
        "sku_names": sku_names,
        "image_size": AE_SIZE,
        "score_quantile": 0.995,
        "calibration_quantile": calibration_quantile,
        "products": {},
    }

    for sku in sku_names:
        train_ds = ROIAnomalyDataset(Path(anomaly_root) / sku / "train" / "good")
        calibration_ds = ROIAnomalyDataset(Path(anomaly_root) / sku / "calibration" / "good")
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(device == "cuda"),
        )
        calibration_loader = DataLoader(
            calibration_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=(device == "cuda"),
        )

        model = SurfaceConvAutoencoder().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for epoch in range(ae_epochs):
            model.train()
            epoch_losses = []
            for images in train_loader:
                images = images.to(device, non_blocking=True)
                reconstructed = model(images)
                loss = 0.7 * F.l1_loss(reconstructed, images) + 0.3 * F.mse_loss(reconstructed, images)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            if not epoch_losses:
                raise RuntimeError(f"{sku} için eğitim batch'i üretilemedi.")
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  {sku} | epoch {epoch + 1}/{ae_epochs} | loss={np.mean(epoch_losses):.6f}")

        calibration_scores = _batch_anomaly_scores(model, calibration_loader, device)
        threshold = float(np.percentile(calibration_scores, calibration_quantile))
        if not np.isfinite(threshold) or threshold <= 0:
            raise RuntimeError(f"{sku} için geçersiz anomali eşiği: {threshold}")

        onnx_name = f"anomali_{sku}.onnx"
        onnx_path = models_path / onnx_name
        model.eval()
        dummy = torch.randn(1, 3, AE_SIZE, AE_SIZE, device=device)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            opset_version=17,
            input_names=["input"],
            output_names=["reconstructed"],
            dynamic_axes={"input": {0: "batch"}, "reconstructed": {0: "batch"}},
            dynamo=False,
        )
        # ONNX dosyasının açılabildiğini hemen doğrula.
        test_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if not test_session.get_inputs():
            raise RuntimeError(f"ONNX girişi bulunamadı: {onnx_path}")

        registry["products"][sku] = {
            "onnx_model": onnx_name,
            "anomaly_threshold": threshold,
            "calibration_samples": len(calibration_scores),
        }
        print(f"  -> {sku} ONNX hazır | eşik={threshold:.6f}")

    registry_path = models_path / "sku_labels.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4, ensure_ascii=False)
    return str(best_weights)


# =====================================================================
# BÖLÜM 3: YOLO + ONNX ÇIKARIM / AYIKLAMA
# =====================================================================
class IntegratedYOLOAnomalySorter:
    def __init__(
        self,
        yolo_weights: str,
        models_dir: str,
        yolo_conf: float = 0.60,
        yolo_iou: float = 0.50,
    ):
        if not 0 < yolo_conf < 1:
            raise ValueError("yolo_conf 0 ile 1 arasında olmalıdır.")
        self.yolo = YOLO(yolo_weights)
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        models_path = Path(models_dir)
        registry_path = models_path / "sku_labels.json"
        if not registry_path.exists():
            raise FileNotFoundError(f"SKU kayıt dosyası bulunamadı: {registry_path}")
        with open(registry_path, "r", encoding="utf-8") as f:
            self.registry = json.load(f)

        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.ae_sessions: dict[str, dict[str, Any]] = {}
        for sku, info in self.registry["products"].items():
            onnx_path = models_path / info["onnx_model"]
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX model bulunamadı: {onnx_path}")
            session = ort.InferenceSession(str(onnx_path), providers=providers)
            self.ae_sessions[sku] = {
                "sess": session,
                "in": session.get_inputs()[0].name,
                "thresh": float(info["anomaly_threshold"]),
            }

        self.class_names = self.yolo.names
        self.valid_skus = set(self.ae_sessions)

    def _inspect_roi(self, crop_bgr: np.ndarray, sku: str) -> tuple[bool, float, float, np.ndarray]:
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Anomali kontrolüne boş ROI gönderildi.")
        if sku not in self.ae_sessions:
            raise KeyError(f"SKU için ONNX modeli yok: {sku}")

        pack = self.ae_sessions[sku]
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        resized = cv2.resize(rgb, (AE_SIZE, AE_SIZE), interpolation=cv2.INTER_AREA)
        tensor = np.transpose(resized, (2, 0, 1))[None].astype(np.float32)
        reconstructed = pack["sess"].run(None, {pack["in"]: tensor})[0]
        recon_img = np.transpose(reconstructed[0], (1, 2, 0))
        error_map = np.mean(np.abs(resized - recon_img), axis=2)
        score = float(np.quantile(error_map, 0.995))
        threshold = pack["thresh"]
        return score <= threshold, score, threshold, error_map

    def process_batch(self, input_dir: str, output_dir: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "TESBIT_YOK").mkdir(exist_ok=True)
        (output_path / "KONTROL_GEREKLI").mkdir(exist_ok=True)
        for sku in self.valid_skus:
            (output_path / sku / "OK").mkdir(parents=True, exist_ok=True)
            (output_path / sku / "NOK_HEATMAP").mkdir(parents=True, exist_ok=True)

        image_paths = list_images(input_dir)
        if not image_paths:
            raise FileNotFoundError(f"Giriş klasöründe görüntü bulunamadı: {input_dir}")

        pred_csv = output_path / "yolo_anomali_sonuclar.csv"
        t0 = time.perf_counter()
        print(f"\n[AYIKLAMA] {len(image_paths)} görüntü işleniyor...")

        with open(pred_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Dosya_Adi", "Tahmin_Urun", "YOLO_Guven", "Tahmin_Durum",
                "Anomali_Skoru", "Esik", "ROI_Sonuclari", "Hata",
            ])

            for idx, path in enumerate(image_paths, start=1):
                fname = os.path.basename(path)
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                if image is None:
                    writer.writerow([fname, "OKUNAMADI", 0, "KONTROL_GEREKLI", "", "", "[]", "Görüntü okunamadı"])
                    continue

                try:
                    result = self.yolo.predict(
                        image, conf=self.yolo_conf, iou=self.yolo_iou, verbose=False
                    )[0]
                    if result.boxes is None or len(result.boxes) == 0:
                        shutil.copy2(path, output_path / "TESBIT_YOK" / fname)
                        writer.writerow([fname, "YOK", 0.0, "TESBIT_YOK", "", "", "[]", "YOLO nesne bulamadı"])
                        continue

                    annotated = image.copy()
                    roi_results: list[dict[str, Any]] = []
                    errors: list[str] = []
                    any_nok = False
                    all_checked = True

                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        sku = self.class_names[cls_id]
                        confidence = float(box.conf[0].item())
                        height, width = image.shape[:2]
                        x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
                        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))

                        if sku not in self.ae_sessions or x2 <= x1 or y2 <= y1:
                            all_checked = False
                            errors.append(f"Geçersiz ROI veya SKU modeli yok: {sku}")
                            continue

                        crop = image[y1:y2, x1:x2]
                        if crop.size == 0:
                            all_checked = False
                            errors.append(f"Boş ROI: {sku}")
                            continue

                        try:
                            is_ok, score, threshold, error_map = self._inspect_roi(crop, sku)
                        except Exception as exc:
                            all_checked = False
                            errors.append(f"{sku} anomali kontrolü: {exc}")
                            continue

                        status = "OK" if is_ok else "NOK"
                        any_nok = any_nok or (not is_ok)
                        roi_results.append({
                            "sku": sku, "confidence": round(confidence, 5),
                            "status": status, "score": round(score, 6),
                            "threshold": round(threshold, 6),
                            "box": [x1, y1, x2, y2],
                        })

                        color = (0, 190, 0) if is_ok else (0, 0, 255)
                        label = f"{sku} {status} {score:.3f}/{threshold:.3f}"
                        if not is_ok:
                            roi = annotated[y1:y2, x1:x2]
                            err_resized = cv2.resize(error_map, (roi.shape[1], roi.shape[0]))
                            norm_error = np.clip(err_resized / max(threshold * 1.5, 1e-6), 0, 1)
                            heatmap = cv2.applyColorMap((norm_error * 255).astype(np.uint8), cv2.COLORMAP_JET)
                            blend = cv2.addWeighted(roi, 0.55, heatmap, 0.45, 0)
                            mask = norm_error > 0.35
                            roi[mask] = blend[mask]
                            annotated[y1:y2, x1:x2] = roi

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            annotated, label, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                        )

                    if not all_checked or not roi_results:
                        final_status = "KONTROL_GEREKLI"
                    elif any_nok:
                        final_status = "NOK"
                    else:
                        final_status = "OK"

                    # Raporun ana SKU'su en yüksek YOLO güvenine sahip tespittir.
                    primary = max(roi_results, key=lambda r: r["confidence"]) if roi_results else None
                    primary_sku = primary["sku"] if primary else "BELIRSIZ"
                    primary_conf = primary["confidence"] if primary else 0.0
                    scores = [r["score"] for r in roi_results]
                    thresholds = [r["threshold"] for r in roi_results]
                    score_text = json.dumps(scores, ensure_ascii=False)
                    threshold_text = json.dumps(thresholds, ensure_ascii=False)
                    roi_json = json.dumps(roi_results, ensure_ascii=False)

                    if final_status == "OK" and primary_sku in self.valid_skus:
                        shutil.copy2(path, output_path / primary_sku / "OK" / fname)
                    elif final_status == "NOK" and primary_sku in self.valid_skus:
                        safe_imwrite(output_path / primary_sku / "NOK_HEATMAP" / f"NOK_{fname}", annotated)
                    else:
                        safe_imwrite(output_path / "KONTROL_GEREKLI" / fname, annotated)

                    writer.writerow([
                        fname, primary_sku, round(primary_conf, 5), final_status,
                        score_text, threshold_text, roi_json, "; ".join(errors),
                    ])

                except Exception as exc:
                    writer.writerow([fname, "HATA", 0.0, "KONTROL_GEREKLI", "", "", "[]", str(exc)])
                    try:
                        safe_imwrite(output_path / "KONTROL_GEREKLI" / fname, image)
                    except Exception:
                        pass

                if idx % 500 == 0:
                    print(f"  -> {idx}/{len(image_paths)} görüntü işlendi...")

        elapsed = time.perf_counter() - t0
        print(f"[TAMAMLANDI] {len(image_paths)} görüntü {elapsed:.2f} sn içinde işlendi.")
        return str(pred_csv)


# =====================================================================
# BÖLÜM 4: DEĞERLENDİRME
# =====================================================================
def evaluate_yolo_anomaly_pipeline(gt_csv: str, pred_csv: str) -> dict[str, Any]:
    gt_map: dict[str, tuple[str, str]] = {}
    with open(gt_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            gt_map[row["Dosya_Adi"]] = (row["Gercek_Urun"], row["Gercek_Durum"])

    pred_map: dict[str, dict[str, str]] = {}
    with open(pred_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pred_map[row["Dosya_Adi"]] = row

    if not gt_map:
        raise ValueError("Ground-truth CSV boş.")
    if not pred_map:
        raise ValueError("Tahmin CSV boş.")

    common = sorted(set(gt_map) & set(pred_map))
    missing_predictions = sorted(set(gt_map) - set(pred_map))
    unknown_predictions = sorted(set(pred_map) - set(gt_map))

    sku_correct = 0
    detection_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for fname in common:
        real_sku, real_status = gt_map[fname]
        pred = pred_map[fname]
        predicted_sku = pred.get("Tahmin_Urun", "")
        predicted_status = pred.get("Tahmin_Durum", "")

        if predicted_status not in {"TESBIT_YOK", "KONTROL_GEREKLI"} and predicted_sku not in {"YOK", "BELIRSIZ", "HATA"}:
            detection_count += 1
            sku_correct += int(real_sku == predicted_sku)

        # Anomali metriğinde algılanmayan ürünler dışarıda bırakılır; ayrıca kaçırma sayısı raporlanır.
        if predicted_status in {"OK", "NOK"}:
            y_true.append(1 if real_status == "NOK" else 0)
            y_pred.append(1 if predicted_status == "NOK" else 0)

    if y_true:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
    else:
        tn = fp = fn = tp = 0
        precision = recall = f1 = fpr = specificity = 0.0

    not_detected = sum(
        pred_map[f].get("Tahmin_Durum") == "TESBIT_YOK" for f in common
    )
    review_required = sum(
        pred_map[f].get("Tahmin_Durum") == "KONTROL_GEREKLI" for f in common
    )
    detection_rate = detection_count / len(common) if common else 0.0
    sku_accuracy = sku_correct / detection_count if detection_count else 0.0

    report = {
        "ground_truth_total": len(gt_map),
        "predictions_total": len(pred_map),
        "matched_files": len(common),
        "missing_predictions": len(missing_predictions),
        "unknown_predictions": len(unknown_predictions),
        "detection_rate": detection_rate,
        "sku_accuracy_on_detected": sku_accuracy,
        "not_detected": not_detected,
        "review_required": review_required,
        "anomaly_evaluated": len(y_true),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "false_positive_rate": float(fpr),
        "specificity": float(specificity),
    }

    print("\n" + "=" * 62)
    print("YOLO + SKU BAZLI ANOMALİ DEĞERLENDİRME RAPORU")
    print("=" * 62)
    print(f"Ground-truth görüntü sayısı       : {len(gt_map)}")
    print(f"Eşleşen tahmin                    : {len(common)}")
    print(f"Eksik tahmin                      : {len(missing_predictions)}")
    print(f"YOLO tespit oranı                 : %{detection_rate * 100:.2f}")
    print(f"Tespit edilenlerde SKU doğruluğu  : %{sku_accuracy * 100:.2f}")
    print(f"Tespit edilemeyen                 : {not_detected}")
    print(f"Manuel kontrol gereken            : {review_required}")
    print(f"Anomali değerlendirmesine giren   : {len(y_true)}")
    print("-" * 62)
    print(f"Precision                         : {precision:.4f}")
    print(f"Recall                            : {recall:.4f}")
    print(f"F1                                : {f1:.4f}")
    print(f"False Positive Rate               : %{fpr * 100:.2f}")
    print(f"Specificity                       : %{specificity * 100:.2f}")
    print(f"TN={tn} | FP={fp} | FN={fn} | TP={tp}")
    print("=" * 62 + "\n")
    return report


# =====================================================================
# ANA ÇALIŞTIRMA
# =====================================================================
def main() -> None:
    set_reproducibility(SEED)
    workspace = Path("yolo_vision_workspace")
    models_dir = workspace / "models"
    output_dir = workspace / "ayiklama_sonuclari"

    env = ConveyorYOLOFactoryEnv(
        base_dir=str(workspace),
        num_skus=15,
        seed=SEED,
    )
    yaml_file = env.build(
        yolo_train_per_sku=60,
        anomaly_train_per_sku=80,
        anomaly_calibration_per_sku=30,
        total_test_images=6000,
        defect_ratio=0.15,
    )

    yolo_weights = train_yolo_and_anomalies(
        yaml_path=yaml_file,
        anomaly_root=str(env.anomaly_train_dir),
        models_dir=str(models_dir),
        yolo_epochs=30,
        ae_epochs=40,
        batch_size=16,
        calibration_quantile=99.5,
    )

    sorter = IntegratedYOLOAnomalySorter(
        yolo_weights=yolo_weights,
        models_dir=str(models_dir),
        yolo_conf=0.60,
        yolo_iou=0.50,
    )
    prediction_csv = sorter.process_batch(
        input_dir=str(env.pool_6000_dir),
        output_dir=str(output_dir),
    )
    report = evaluate_yolo_anomaly_pipeline(
        gt_csv=str(env.gt_csv_path),
        pred_csv=prediction_csv,
    )
    with open(output_dir / "performans_raporu.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
