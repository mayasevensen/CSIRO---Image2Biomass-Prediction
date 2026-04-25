# DA-Fusion Augmentation - Ragnhild's Contribution

## Method
Adaptation of DA-Fusion (Trabucco et al., ICLR 2024) to the 
Image2Biomass regression task.

## Contribution included:
- Implemented textual inversion pipeline for pasture species tokens
- Generated synthetic images using Stable Diffusion img2img with the learned tokens
- Added 1071 synthetic labeled training samples
- Designed fold-safe synthetic filtering to prevent leakage
- Integrated synthetic data into DINOv2 training pipeline

## Files
- textual_inversion_train.py: trains one token per species
- generate_augmented.py: generates synthetic images via img2img
- da_fusion_pipeline.ipynb: comparison experiment (3 conditions)
- kaggle_da_fusion_pipeline.ipynb: same as above but compatible with Kaggle environment
- learned_tokens/: 9 trained token embeddings

## How to reproduce
1. Run textual_inversion_train.py on a GPU 
2. Run generate_augmented.py to generate synthetic images
3. Run da_fusion_pipeline.ipynb locally or kaggle_da_fusion_pipeline.ipynb on Kaggle to compare methods

More information on the implementation and results can be found in the respective notebooks.