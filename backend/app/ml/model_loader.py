# model_loader.py — saved model ko ek baar load karta hai startup pe
import torch
import torch.nn as nn
from torchvision import models
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(weights_path, num_classes):
    m = models.resnet18()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    return m.eval().to(DEVICE)

# Yeh line tabhi chalegi jab .pth file exist kare
MODEL_PATH = "ml_training/saved_models/retina_v1.pth"

if os.path.exists(MODEL_PATH):
    retina_model = load_model(MODEL_PATH, num_classes=5)
else:
    retina_model = None
    print("⚠️  retina_v1.pth nahi mili — pehle Kaggle pe train karo")