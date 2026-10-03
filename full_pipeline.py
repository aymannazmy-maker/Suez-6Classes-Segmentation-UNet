import torch
import numpy as np
from PIL import Image
import rasterio
import os
from stage1_landcover import LandCoverUNet
from stage2_signal import PropagationUNet

def run_full_pipeline(sat_path='data/raw/suez.jpg', dem_path='data/raw/suez_dem.tif'):
    print("=== Suez 2-Stage Pipeline ===")
    # 1. Stage 1: RGB -> 6 Classes
    print("[Stage1] Segmenting landcover...")
    img = np.array(Image.open(sat_path).convert('RGB'))
    h,w,_ = img.shape

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        dem = np.array(Image.fromarray(dem).resize((w,h))).astype(np.float32)

    # Resize to 256 for model
    img_pil = Image.fromarray(img).resize((256,256))
    dem_pil = Image.fromarray(dem).resize((256,256))

    img_t = torch.from_numpy(np.array(img_pil)).permute(2,0,1).unsqueeze(0).float()/255.0
    dem_t = torch.from_numpy(np.array(dem_pil)).unsqueeze(0).unsqueeze(0).float()
    dem_t = (dem_t - dem_t.min()) / (dem_t.max()-dem_t.min()+1e-6)

    stage1 = LandCoverUNet(6)
    stage1.eval()
    with torch.no_grad():
        pred1 = stage1(img_t)
        landcover = torch.argmax(pred1, dim=1) # [1,256,256]

    # One-hot 6 channels
    landcover_onehot = torch.nn.functional.one_hot(landcover, num_classes=6).permute(0,3,1,2).float() # [1,6,256,256]

    # 2. Stage 2: Build 10 channel input
    print("[Stage2] Building 10-channel input...")
    # [RGB(3) + OneHot(6) + DEM(1)] = 10
    input_10ch = torch.cat([img_t, landcover_onehot, dem_t], dim=1) # [1,10,256,256]
    print(f"Input shape: {input_10ch.shape} -> 3 RGB + 6 Landcover + 1 DEM = 10 channels")

    stage2 = PropagationUNet(in_channels=10, out_channels=16)
    stage2.eval()
    with torch.no_grad():
        signal_map = stage2(input_10ch) # [1,16,256,256]
        signal_class = torch.argmax(signal_map, dim=1).squeeze(0).numpy()

    # حفظ النتايج
    os.makedirs('results', exist_ok=True)
    Image.fromarray((landcover.squeeze(0).numpy()*42).astype(np.uint8)).save('results/stage1_landcover.png')
    Image.fromarray((signal_class*16).astype(np.uint8)).save('results/stage2_signal.png')

    print("Saved: results/stage1_landcover.png (6 classes)")
    print("Saved: results/stage2_signal.png (16 signal levels)")
    print("Done! This is exactly the paper pipeline.")
    return signal_class

if __name__ == "__main__":
    run_full_pipeline()
