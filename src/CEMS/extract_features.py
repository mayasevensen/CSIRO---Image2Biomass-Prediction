"""
Feature extraction for CEMS pipeline.

Preprocessing settings used to generate the cache (changing any of these
requires deleting the cache files and re-running):
  DINOv2:
    model   : facebookresearch/dinov2  dinov2_vits14  (frozen)
    resize  : 504 x 252  (width x height), preserving 2:1 aspect ratio
    patches : 36 x 18 = 648 tokens  (patch_size=14)
    output  : CLS token  →  384-d
    norm    : ImageNet mean/std  [0.485,0.456,0.406] / [0.229,0.224,0.225]

  ResNet50:
    model   : torchvision ResNet50 (ImageNet pretrained, frozen, fc removed)
    crop    : left 1000x1000 + right 1000x1000 halves of the 2000x1000 image
    resize  : 224 x 224 per half
    output  : average of two 2048-d GAP vectors
    norm    : ImageNet mean/std

Cache files written to src/CEMS/cache/:
  features_dinov2.npy   (N, 384)
  features_resnet50.npy (N, 2048)
  image_ids.npy         (N,)   string array of image IDs
"""

import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = REPO_ROOT / "data" / "image" / "train"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
TEST_IMAGE_DIR = REPO_ROOT / "data" / "image" / "test"

DINOV2_CACHE = CACHE_DIR / "features_dinov2.npy"
RESNET_CACHE = CACHE_DIR / "features_resnet50.npy"
IDS_CACHE = CACHE_DIR / "image_ids.npy"
TEST_IMAGE_CACHE = CACHE_DIR / "test_features_dinov2.npy"
TEST_IDS_CACHE = CACHE_DIR / "test_image_ids.npy"


# ---------------------------------------------------------------------------
# DINOv2 extractor
# ---------------------------------------------------------------------------

DINOV2_RESIZE = (504, 252)  # (width, height) → 36×18 patches at patch_size=14

_dinov2_transform = transforms.Compose([
    transforms.Resize((DINOV2_RESIZE[1], DINOV2_RESIZE[0])),  # H, W
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _load_dinov2(device):
    print("Loading DINOv2 ViT-S/14 from torch.hub (frozen)...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def _smoke_test_dinov2(model, device):
    """Verify the model accepts 504x252 input and returns a 384-d CLS token."""
    dummy = torch.zeros(1, 3, DINOV2_RESIZE[1], DINOV2_RESIZE[0]).to(device)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 384), f"DINOv2 smoke test failed: got shape {out.shape}"
    print(f"  DINOv2 smoke test passed — output shape: {out.shape}")


def extract_dinov2(image_paths, device):
    """Return (N, 384) array of CLS-token features."""
    model = _load_dinov2(device)
    _smoke_test_dinov2(model, device)

    feats = []
    for i, p in enumerate(image_paths):
        img = Image.open(p).convert("RGB")
        x = _dinov2_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(x).squeeze(0).cpu().numpy()
        feats.append(feat)
        if (i + 1) % 50 == 0:
            print(f"  DINOv2: {i + 1}/{len(image_paths)}")
    return np.stack(feats)  # (N, 384)


# ---------------------------------------------------------------------------
# ResNet50 extractor  (split-and-average, copied from src/models/resnet.py)
# ---------------------------------------------------------------------------

_resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _load_resnet50(device):
    print("Loading ResNet50 (frozen, fc removed)...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(resnet.children())[:-1])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def _resnet_single(model, img_pil, device):
    """2048-d feature for one PIL image after standard ResNet preprocessing."""
    t = _resnet_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(t).squeeze().cpu().numpy()
    return feat


def extract_resnet50(image_paths, device):
    """Return (N, 2048) array via split-and-average on 2000x1000 images."""
    model = _load_resnet50(device)

    feats = []
    for i, p in enumerate(image_paths):
        img = Image.open(p).convert("RGB")
        left = img.crop((0, 0, 1000, 1000))
        right = img.crop((1000, 0, 2000, 1000))
        feat = (_resnet_single(model, left, device) +
                _resnet_single(model, right, device)) / 2.0
        feats.append(feat)
        if (i + 1) % 50 == 0:
            print(f"  ResNet50: {i + 1}/{len(image_paths)}")
    return np.stack(feats)  # (N, 2048)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if DINOV2_CACHE.exists() and RESNET_CACHE.exists() and IDS_CACHE.exists() and TEST_IMAGE_CACHE.exists() and TEST_IDS_CACHE.exists():
        print("Cache already exists — skipping extraction.")
        print(f"  DINOv2:  {np.load(DINOV2_CACHE).shape}")
        print(f"  ResNet50:{np.load(RESNET_CACHE).shape}")
        print(f"  Test images: {np.load(TEST_IDS_CACHE).shape}")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    test_image_paths = sorted(TEST_IMAGE_DIR.glob("*.jpg"))
    image_ids = np.array([p.stem for p in image_paths])
    test_image_ids = np.array([p.stem for p in test_image_paths])
    print(f"Found {len(image_paths)} training images.")
    print(f"Found {len(test_image_paths)} test images.")

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n--- DINOv2 extraction ---")
    feats_dino = extract_dinov2(image_paths, device)
    test_feats_dino = extract_dinov2(test_image_paths, device)
    print(f"  Done. Shape: {feats_dino.shape}")
    print(f"  Test images: {test_feats_dino.shape}")

    print("\n--- ResNet50 extraction ---")
    feats_resnet = extract_resnet50(image_paths, device)
    print(f"  Done. Shape: {feats_resnet.shape}")

    np.save(DINOV2_CACHE, feats_dino)
    np.save(RESNET_CACHE, feats_resnet)
    np.save(IDS_CACHE, image_ids)
    np.save(TEST_IMAGE_CACHE, test_feats_dino)
    np.save(TEST_IDS_CACHE, test_image_ids)
    print(f"\nCache saved to {CACHE_DIR}/")


if __name__ == "__main__":
    main()
