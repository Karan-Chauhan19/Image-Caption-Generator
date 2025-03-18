"""
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J University
"""

import os
import torch
import pickle
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from config.config import Config
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights

class VGG16:
    def __init__(self, device):
        self.device = device
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(
            *vgg.features,
            vgg.avgpool,
            nn.Flatten(),
            *vgg.classifier[:-1]
        ).to(device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_dir):
        features = {}
        with torch.no_grad():
            for img_name in tqdm(os.listdir(image_dir)):
                img_path = os.path.join(image_dir, img_name)
                image = Image.open(img_path).convert('RGB')
                image = self.preprocess(image).unsqueeze(0).to(self.device)
                
                feature = self.model(image)
                image_id = img_name.split('.')[0]
                features[image_id] = feature.cpu().numpy()
        
        feature_path = os.path.join(Config().PATH_TO_SAVE_TRAINED_MODEL, 'features.pkl')
        with open(feature_path, 'wb') as f:
            pickle.dump(features, f)
        return features

    def load_features(self, path= Config().PATH_TO_SAVE_TRAINED_MODEL+'features.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)