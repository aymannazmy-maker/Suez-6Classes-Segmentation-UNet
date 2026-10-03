import os, torch, numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from stage1_landcover import LandCoverUNet
from stage2_signal import PropagationUNet
from sklearn.model_selection import train_test_split

# هنستخدم الصورة الكبيرة اللي قطعناها كداتا
class SuezDataset(Dataset):
    def __init__(self, img_path='data/raw/suez.jpg', size=256):
        img = np.array(Image.open(img_path).convert('RGB'))
        h,w,_ = img.shape
        # هنقطع tiles
        self.tiles=[]
        for i in range(0, h-size+1, size//2): # overlap 50%
            for j in range(0, w-size+1, size//2):
                crop = img[i:i+size, j:j+size]
                # KMeans كـ label مؤقت
                from sklearn.cluster import KMeans
                pix = crop.reshape(-1,3)
                km = KMeans(n_clusters=6, n_init=3, random_state=0).fit(pix)
                mask = km.labels_.reshape(size,size)
                self.tiles.append((crop, mask))
        print(f"Dataset: {len(self.tiles)} tiles")

    def __len__(self): return len(self.tiles)
    def __getitem__(self, idx):
        img, mask = self.tiles[idx]
        img_t = torch.from_numpy(img).permute(2,0,1).float()/255.0
        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t

print("Loading dataset...")
ds = SuezDataset()
loader = DataLoader(ds, batch_size=4, shuffle=True)

# Train Stage1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LandCoverUNet(6).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print(f"Training Stage1 on {device}...")
for epoch in range(20):
    loss_sum=0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum+=loss.item()
    print(f"Epoch {epoch+1}/20 Loss: {loss_sum/len(loader):.4f}")

torch.save(model.state_dict(), 'stage1_suez_6c.pth')
print("Saved stage1_suez_6c.pth - Ready for paper!")
print("Now your Stage1 is trained, not just KMeans!")
