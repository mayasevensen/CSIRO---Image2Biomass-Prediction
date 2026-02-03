# features.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class ResNetExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        print(f"Loading ResNet50 on {self.device}...")
        
        # Load standard ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Remove the final classification layer (fc) to get raw features (2048 size)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def get_features(self, image_path):
        """
        Input: Path to a 2000x1000 image.
        Output: A 2048-dimensional feature vector.
        Method: Splits image into two 1000x1000 halves, processes both, and averages them.
        """
        if not os.path.exists(image_path):
            print(f"Warning: File not found {image_path}")
            return np.zeros(2048)

        try:
            img = Image.open(image_path).convert('RGB')
            
            # 1. Crop the left and right halves (1000x1000 each)
            img_left = img.crop((0, 0, 1000, 1000))
            img_right = img.crop((1000, 0, 2000, 1000))
            
            # 2. Preprocess
            t_left = self.preprocess(img_left)
            t_left = t_left.unsqueeze(0).to(self.device)
            t_right = self.preprocess(img_right)
            t_right = t_right.unsqueeze(0).to(self.device)
            
            # 3. Extract Features
            with torch.no_grad():
                feat_left = self.model(t_left).squeeze().cpu().numpy()
                feat_right = self.model(t_right).squeeze().cpu().numpy()
            
            # 4. Average the features
            return (feat_left + feat_right) / 2.0
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return np.zeros(2048)