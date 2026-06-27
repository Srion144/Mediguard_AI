# preprocess.py — resize and normalize image for model input
from torchvision import transforms

_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def preprocess(pil_img):
    # add batch dimension: (1, 3, 224, 224)
    return _tf(pil_img.convert("RGB")).unsqueeze(0)