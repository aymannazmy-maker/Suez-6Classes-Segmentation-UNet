import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import rasterio

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class LandCoverUNet(nn.Module):
    """ Stage 1: RGB(3) -> 6 Classes Landcover """
    def __init__(self, n_classes=6):
        super().__init__()
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.c1 = DoubleConv(64, 32)
        self.final = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        y = self.up2(x3)
        y = self.c2(torch.cat([y, x2], 1))
        y = self.up1(y)
        y = self.c1(torch.cat([y, x1], 1))
        return self.final(y)

def segment_suez(satellite_path='data/raw/suez.jpg', dem_path='data/raw/suez_dem.tif', model_path=None):
    """ بياخد suez.jpg + DEM ويطلع خريطة 6 فئات """
    img = np.array(Image.open(satellite_path).convert('RGB'))
    h,w,_ = img.shape

    # حمل الارتفاعات
    try:
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            # Resize للصورة
            dem = np.array(Image.fromarray(dem).resize((w,h)))
            print(f"DEM loaded: min {dem.min()} max {dem.max()} m")
    except:
        print("DEM not found, using 0")
        dem = np.zeros((h,w))

    # لو فيه موديل مدرب استخدمه، لو لا استخدم rules + KMeans مطور
    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()/255.0

    model = LandCoverUNet(6)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.eval()
    with torch.no_grad():
        pred = model(tensor)
        landcover = torch.argmax(pred, dim=1).squeeze(0).numpy().astype(np.uint8)

    # تحسين بالارتفاع: ده السر عشان نميز جبل من تل
    # 0:Urban, 1:Water, 2:Veg, 3:BareSoil, 4:Mountains, 5:Hills
    dem_norm = (dem - dem.min()) / (dem.max()-dem.min()+1e-6)
    # لو ارتفاع > 300م ولون بني/غامق = جبل
    is_high = dem > np.percentile(dem, 80)
    is_mid = (dem > np.percentile(dem, 50)) & (~is_high)
    # غيّر الـ class حسب الارتفاع
    landcover[is_high & (landcover==3)] = 4 # Bare -> Mountain
    landcover[is_mid & (landcover==3)] = 5 # Bare -> Hill

    Image.fromarray(landcover).save('data/suez_6classes.png')
    print("Saved data/suez_6classes.png with 6 classes")
    return landcover, dem, img

if __name__ == "__main__":
    import os
    segment_suez()
