"""
DA-Fusion Step 2: Generate Augmented Images
============================================
For each training image:
  - Group A species (token learned): uses <species_token> + img2img
  - Group B species (too few images): uses generic prompt + img2img

USAGE:
    python generate_augmented.py \
        --train_csv path/to/data/tabular/train/train.csv \
        --image_dir path/to/data/image \
        --token_dir ./learned_tokens \
        --output_image_dir path/to/data/image/augmented \
        --output_csv path/to/data/tabular/train/train_augmented.csv \
        --augmentations_per_image 3 \
        --strength 0.4
"""

import os
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

# ─── Config ────────────────────────────────────────────────────────────────────

MODEL_ID = "runwayml/stable-diffusion-v1-5"
SEED = 42

# Species that did NOT get textual inversion (too few images)
GENERIC_SPECIES = {"Mixed", "SubcloverDalkeith", "SubcloverLosa",
                   "Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass"}

# Generic prompt used for species without a learned token
# Deliberately vague to avoid data leakage (matches paper's data-centric approach)
GENERIC_PROMPT = "a photograph of pasture vegetation, outdoor field, natural lighting"

# ─── Token name helper (must match textual_inversion_train.py) ─────────────────

def species_to_token(species: str) -> str:
    safe = species.replace("_", "").replace(" ", "").lower()
    return f"<{safe}>"

# ─── Load pipeline with injected token embeddings ──────────────────────────────

def load_pipeline_with_tokens(token_dir: str, device: str):
    """
    Loads the SD img2img pipeline and injects all learned token embeddings
    into the text encoder at once. This avoids reloading the model per species.
    """
    print("Loading Stable Diffusion img2img pipeline...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,        # Disable - field photos won't trigger it anyway
    ).to(device)

    token_map = {}   # species -> token string

    token_files = list(Path(token_dir).glob("*.pt"))
    if not token_files:
        print("  [WARN] No learned tokens found in token_dir. Will use generic prompts for all.")
        return pipe, token_map

    print(f"  Injecting {len(token_files)} learned token(s)...")
    for token_file in token_files:
        data = torch.load(token_file, map_location="cpu")
        species = data["species"]
        token = data["token"]
        embedding = data["embedding"]   # shape: (hidden_size,)

        # Add token to tokenizer
        pipe.tokenizer.add_tokens([token])
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        token_id = pipe.tokenizer.convert_tokens_to_ids(token)

        with torch.no_grad():
            pipe.text_encoder.get_input_embeddings().weight[token_id] = (
                embedding.to(pipe.text_encoder.dtype).to(device)
            )

        token_map[species] = token
        print(f"    Injected: {token}  ({species})")

    pipe.text_encoder.eval()
    return pipe, token_map

# ─── Build prompt for a given species ──────────────────────────────────────────

def build_prompt(species: str, token_map: dict) -> str:
    if species in GENERIC_SPECIES or species not in token_map:
        return GENERIC_PROMPT
    token = token_map[species]
    templates = [
        f"a field photograph of {token} pasture vegetation, natural lighting",
        f"an outdoor photo of {token} grass, realistic, daylight",
        f"a top-down image of {token} biomass in a paddock",
        f"a realistic photo of {token} vegetation, field setting",
    ]
    return random.choice(templates)

# ─── Augment one image ─────────────────────────────────────────────────────────

def augment_image(pipe, image_path: str, prompt: str,
                  strength: float, guidance_scale: float,
                  aug_idx: int) -> Image.Image:
    """
    Load one field image, randomly pick left or right half,
    run img2img with the given prompt.
    """
    img = Image.open(image_path).convert("RGB")

    # Pick a half - vary by aug_idx for diversity across augmentations
    if aug_idx % 2 == 0:
        crop = img.crop((0, 0, 1000, 1000))
    else:
        crop = img.crop((1000, 0, 2000, 1000))

    # Resize to 512x512 for SD
    crop = crop.resize((512, 512), Image.LANCZOS)

    result = pipe(
        prompt=prompt,
        image=crop,
        strength=strength,          # 0.3–0.5: lower = closer to original
        guidance_scale=guidance_scale,
        num_inference_steps=30,
    ).images[0]

    return result

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--image_dir", required=True,
                        help="Root folder containing train/ subfolder")
    parser.add_argument("--token_dir", required=True,
                        help="Folder with .pt files from textual_inversion_train.py")
    parser.add_argument("--output_image_dir", required=True,
                        help="Where to save generated images (e.g. data/image/augmented)")
    parser.add_argument("--output_csv", required=True,
                        help="Output CSV path for augmented dataset")
    parser.add_argument("--augmentations_per_image", type=int, default=3,
                        help="How many augmented versions to generate per real image")
    parser.add_argument("--strength", type=float, default=0.4,
                        help="img2img strength (0=no change, 1=ignore original). 0.3-0.5 recommended.")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--skip_real", action="store_true",
                        help="If set, output CSV contains ONLY synthetic images (not real ones)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.output_image_dir, exist_ok=True)

    # Load training CSV
    df = pd.read_csv(args.train_csv)

    # Get unique images with their species and targets (wide format)
    df_unique = df.drop_duplicates(subset="image_path").copy()

    # Pivot to get all 5 targets per image
    targets = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    df_pivot = df.pivot_table(
        index=["image_path", "Species", "Sampling_Date", "State", "Height_Ave_cm", "Pre_GSHH_NDVI"],
        columns="target_name",
        values="target"
    ).reset_index()

    print(f"Found {len(df_pivot)} unique training images across "
          f"{df_pivot['Species'].nunique()} species.")

    # Load pipeline + inject learned tokens
    pipe, token_map = load_pipeline_with_tokens(args.token_dir, device)

    # ── Generate augmented images ──────────────────────────────────────────────
    synthetic_rows = []
    total = len(df_pivot)

    for img_idx, row in df_pivot.iterrows():
        rel_path = row["image_path"]                  # e.g. "train/ID123.jpg"
        full_path = os.path.join(args.image_dir, rel_path)
        species = row["Species"]

        if not os.path.exists(full_path):
            print(f"  [WARN] Image not found: {full_path}")
            continue

        prompt = build_prompt(species, token_map)

        if (img_idx + 1) % 50 == 0 or img_idx == 0:
            print(f"  [{img_idx+1}/{total}] {rel_path} | species: {species}")
            print(f"    prompt: {prompt[:80]}...")

        for aug_i in range(args.augmentations_per_image):
            # Vary strength slightly per augmentation for diversity
            strength_noise = random.uniform(-0.05, 0.05)
            actual_strength = max(0.2, min(0.6, args.strength + strength_noise))

            try:
                aug_img = augment_image(pipe, full_path, prompt,
                                        actual_strength, args.guidance_scale, aug_i)
            except Exception as e:
                print(f"  [ERROR] Failed to augment {rel_path} aug {aug_i}: {e}")
                continue

            # Save augmented image
            base_name = Path(rel_path).stem    # e.g. "ID123"
            aug_filename = f"{base_name}_aug{aug_i}.jpg"
            aug_save_path = os.path.join(args.output_image_dir, aug_filename)
            aug_img.save(aug_save_path, quality=95)

            # Build a new row with the same targets as the source image
            # (label preservation: synthetic image inherits real image's labels)
            synthetic_row = {
                "image_path": os.path.join("augmented", aug_filename),
                "Species": species,
                "Sampling_Date": row["Sampling_Date"],
                "State": row["State"],
                "Height_Ave_cm": row["Height_Ave_cm"],
                "Pre_GSHH_NDVI": row["Pre_GSHH_NDVI"],
                "is_synthetic": True,
                "source_image": rel_path,
                "aug_strength": actual_strength,
                "prompt_used": prompt,
            }
            for t in targets:
                synthetic_row[t] = row[t]

            synthetic_rows.append(synthetic_row)

    # ── Build output CSV ───────────────────────────────────────────────────────
    df_synthetic = pd.DataFrame(synthetic_rows)

    if args.skip_real:
        df_out = df_synthetic
    else:
        # Add real images too (with is_synthetic=False)
        df_real = df_pivot.copy()
        df_real["is_synthetic"] = False
        df_real["source_image"] = df_real["image_path"]
        df_real["aug_strength"] = 0.0
        df_real["prompt_used"] = ""
        df_out = pd.concat([df_real, df_synthetic], ignore_index=True)

    df_out.to_csv(args.output_csv, index=False)

    print(f"\nDone!")
    print(f"  Real images:      {len(df_pivot)}")
    print(f"  Synthetic images: {len(df_synthetic)}")
    print(f"  Total in CSV:     {len(df_out)}")
    print(f"  Saved to:         {args.output_csv}")
    print(f"\nNext step: use da_fusion_pipeline.ipynb locally with the new CSV.")

if __name__ == "__main__":
    main()