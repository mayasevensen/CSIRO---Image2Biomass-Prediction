# DA-Fusion Augmentation - Ragnhild's Contribution

## Method
Adaptation of DA-Fusion (Trabucco et al., ICLR 2024) to the 
Image2Biomass regression task.

## Files
- textual_inversion_train.py  - trains one token per species
- generate_augmented.py       - generates synthetic images via img2img
- da_fusion_pipeline.ipynb    - comparison experiment (3 conditions)
- learned_tokens/             - 9 trained token embeddings

## How to reproduce
1. Run textual_inversion_train.py on a GPU 
2. Run generate_augmented.py to generate synthetic images
3. Run da_fusion_pipeline.ipynb locally

## Results
| Method       | Weighted R² |
|--------------|-------------|
| Baseline     | 0.6273      |
| RandAugment  | 0.5193      |
| DA-Fusion    | 0.6202      |