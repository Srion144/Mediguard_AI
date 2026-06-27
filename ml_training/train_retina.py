# train_retina.py — Kaggle notebook mein yeh run karo
import os, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_DIR  = "/kaggle/input/aptos2019-blindness-detection"
IMG_DIR   = os.path.join(DATA_DIR, "train_images")
CSV       = os.path.join(DATA_DIR, "train.csv")
NUM_CLASSES = 5
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

class RetinaDataset(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row  = self.df.iloc[i]
        path = os.path.join(IMG_DIR, row.id_code + ".png")
        img  = Image.open(path).convert("RGB")
        return self.tf(img), int(row.diagnosis)

norm = transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])

tf_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(), norm,
])
tf_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), norm,
])

df = pd.read_csv(CSV)
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df.diagnosis, random_state=42)

train_loader = DataLoader(RetinaDataset(train_df, tf_train),
                          batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(RetinaDataset(val_df, tf_val),
                          batch_size=32, num_workers=2)

# Transfer learning — pretrained ResNet18, sirf last layer badlo
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print(f"Training on: {DEVICE}")
best_acc = 0.0

for epoch in range(10):
    # --- Training ---
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()

    # --- Validation ---
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1:02d}  val_acc = {acc:.3f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "retina_v1.pth")
        print(f"  ✅ Best model saved! acc={best_acc:.3f}")

print(f"\nTraining complete. Best val_acc = {best_acc:.3f}")
print("retina_v1.pth download kar lo Kaggle se!")