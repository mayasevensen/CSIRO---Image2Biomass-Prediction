"""
DA-Fusion Step 1: Textual Inversion Training
============================================
For each species with >= 10 images, learns a custom token <species_token>
that represents the visual appearance of that species in your dataset.

Species with < 10 images (Mixed, SubcloverDalkeith, SubcloverLosa,
Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass) are skipped - they will
use generic prompts during augmentation instead.

USAGE:
    python textual_inversion_train.py \
        --train_csv path/to/data/tabular/train/train.csv \
        --image_dir path/to/data/image \
        --output_dir ./learned_tokens
"""

import os
import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.backends.cuda.matmul.allow_tf32 = True

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# ─── Config ────────────────────────────────────────────────────────────────────

MODEL_ID = "runwayml/stable-diffusion-v1-5"
MIN_IMAGES_FOR_TI = 10          # Species below this threshold use generic prompts
NUM_TRAIN_STEPS = 500           
LEARNING_RATE = 5e-4
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
IMAGE_SIZE = 512                # SD default
SEED = 42

# Species with < MIN_IMAGES_FOR_TI - will be handled by generic prompts
SKIP_SPECIES = {"Mixed", "SubcloverDalkeith", "SubcloverLosa",
                "Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass"}

# ─── Token name helper ─────────────────────────────────────────────────────────

def species_to_token(species: str) -> str:
    """Convert species name to a safe placeholder token."""
    safe = species.replace("_", "").replace(" ", "").lower()
    return f"<{safe}>"

# ─── Dataset ───────────────────────────────────────────────────────────────────

class SpeciesDataset(Dataset):
    """
    Returns (image_tensor, prompt) pairs for one species.
    Images are the 2000x1000 field photos, we randomly crop one
    of the two 1000x1000 halves per sample to get diversity.
    """
    def __init__(self, image_paths, token, tokenizer, image_size=512):
        self.image_paths = image_paths
        self.token = token
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.templates = [
            f"a photo of {token} pasture",
            f"a field photograph of {token} grass",
            f"an outdoor image of {token} vegetation",
            f"a top-down view of {token} biomass",
        ]
        self.transform_fn = self._make_transform()

    def _make_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        # Oversample so we get NUM_TRAIN_STEPS even with few images
        return max(len(self.image_paths) * 4, 64)

    def __getitem__(self, idx):
        path = self.image_paths[idx % len(self.image_paths)]
        img = Image.open(path).convert("RGB")

        # Randomly pick left or right half of the 2000x1000 image
        if random.random() < 0.5:
            img = img.crop((0, 0, 1000, 1000))
        else:
            img = img.crop((1000, 0, 2000, 1000))

        img_tensor = self.transform_fn(img)

        prompt = random.choice(self.templates)
        token_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        return {"pixel_values": img_tensor, "input_ids": token_ids}

# ─── Training loop ─────────────────────────────────────────────────────────────

def train_one_species(species, image_paths, output_dir, device, args):
    token = species_to_token(species)
    print(f"\n{'='*60}")
    print(f"  Training token {token} on {len(image_paths)} images")
    print(f"{'='*60}")

    save_path = Path(output_dir) / f"{species}.pt"
    if save_path.exists():
        print(f"  [SKIP] Token already exists at {save_path}")
        return

    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # Add the new token to the tokenizer
    num_added = tokenizer.add_tokens([token])
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(token)

    # Initialise new token embedding from "grass" as a sensible starting point
    grass_id = tokenizer.convert_tokens_to_ids("grass")
    with torch.no_grad():
        text_encoder.get_input_embeddings().weight[token_id] = (
            text_encoder.get_input_embeddings().weight[grass_id].clone()
        )

    # Freeze everything except the new token embedding
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder.get_input_embeddings().weight.requires_grad_(True)

    vae.to(device)
    unet.to(device)
    text_encoder.to(device)

    optimizer = torch.optim.AdamW(
        [text_encoder.get_input_embeddings().weight],
        lr=args.learning_rate,
    )

    dataset = SpeciesDataset(image_paths, token, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader_iter = iter(dataloader)

    global_step = 0
    text_encoder.train()

    for step in range(args.num_steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)

        # Encode images to latent space
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

        # Sample noise and timestep
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (latents.shape[0],), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get text embeddings
        encoder_hidden_states = text_encoder(input_ids)[0]

        # Predict noise
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        loss = loss / args.grad_accum
        loss.backward()

        # Only update the new token — freeze all other embeddings
        grads = text_encoder.get_input_embeddings().weight.grad
        grads_mask = torch.zeros_like(grads)
        grads_mask[token_id] = 1.0
        text_encoder.get_input_embeddings().weight.grad = grads * grads_mask

        if (step + 1) % args.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1
        if global_step % 100 == 0:
            print(f"  Step {global_step}/{args.num_steps} | loss: {loss.item() * args.grad_accum:.4f}")

    # Save only the learned embedding
    learned_embedding = (
        text_encoder.get_input_embeddings().weight[token_id].detach().cpu()
    )
    torch.save({"token": token, "embedding": learned_embedding, "species": species},
               save_path)
    print(f"  Saved token embedding → {save_path}")

    # Free GPU memory before next species
    del vae, unet, text_encoder, noise_scheduler, tokenizer
    torch.cuda.empty_cache()

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", default="./learned_tokens")
    parser.add_argument("--num_steps", type=int, default=NUM_TRAIN_STEPS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--min_images", type=int, default=MIN_IMAGES_FOR_TI)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: CPU training is very slow. Use a GPU instance.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and pivot training CSV
    df = pd.read_csv(args.train_csv)
    # Get unique (image_path, Species) pairs
    df_unique = df.drop_duplicates(subset="image_path")[["image_path", "Species"]]

    # Group by species
    species_groups = df_unique.groupby("Species")["image_path"].apply(list).to_dict()

    print(f"\nFound {len(species_groups)} species:")
    for sp, paths in sorted(species_groups.items(), key=lambda x: len(x[1])):
        status = "SKIP (too few)" if sp in SKIP_SPECIES or len(paths) < args.min_images else "TRAIN"
        print(f"  {sp:<55} {len(paths):>3} images  [{status}]")

    # Train one token per eligible species
    trainable = {sp: paths for sp, paths in species_groups.items()
                 if sp not in SKIP_SPECIES and len(paths) >= args.min_images}

    print(f"\nWill train {len(trainable)} tokens.")

    for species, rel_paths in trainable.items():
        full_paths = [os.path.join(args.image_dir, p) for p in rel_paths]
        full_paths = [p for p in full_paths if os.path.exists(p)]
        if not full_paths:
            print(f"  [WARN] No images found for {species}, skipping.")
            continue
        train_one_species(species, full_paths, args.output_dir, device, args)

    print("\nDone! All tokens saved to:", args.output_dir)
    print("Next step: run generate_augmented.py")

if __name__ == "__main__":
    main()