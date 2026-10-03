import numpy as np
from PIL import Image
import rasterio
from sklearn.cluster import KMeans

# 1. حمل الصورة الأصلية
img = np.array(Image.open('data/raw/suez.jpg').convert('RGB'))
h,w,_ = img.shape

# 2. حمل الـ DEM
with rasterio.open('data/raw/suez_dem.tif') as src:
    dem = src.read(1)
    dem_r = np.array(Image.fromarray(dem).resize((w,h)))

print(f"DEM min {dem_r.min()} max {dem_r.max()}")

# 3. اعمل 6 Classes حقيقي بـ KMeans + DEM
pixels = img.reshape(-1,3)
kmeans = KMeans(n_clusters=6, n_init=5, random_state=42).fit(pixels)
labels = kmeans.labels_.reshape(h,w)

# صحح بالارتفاع: عالي = جبل
high_thr = np.percentile(dem_r, 85)
mid_thr = np.percentile(dem_r, 60)
is_mountain = dem_r > high_thr
is_hill = (dem_r > mid_thr) & (dem_r <= high_thr)

# لو KMeans مطلعه ارض فاضية وارتفاعه عالي حوله لجبل
# نفترض cluster 0,1 هما الارض
labels[is_mountain] = 4 # 4: Mountains
labels[is_hill] = 5 # 5: Hills

# 4. Colormap للـ 6 فئات زي ما اتفقنا
# 0:Urban=احمر, 1:Water=ازرق, 2:Veg=اخضر, 3:Bare=اصفر, 4:Mountains=بني غامق, 5:Hills=بني فاتح
colormap = np.array([
    [200, 50, 50], # 0 Urban
    [ 50, 100, 255], # 1 Water
    [ 50, 200, 50], # 2 Vegetation
    [230, 220, 150], # 3 Bare Soil
    [120, 80, 40], # 4 Mountains
    [180, 150, 100], # 5 Hills
], dtype=np.uint8)

colored = colormap[labels]
Image.fromarray(colored).save('results/stage1_landcover.png')
Image.fromarray((labels*42).astype(np.uint8)).save('results/stage1_labels_gray.png')

# 5. اعمل Signal Map وهمي للعرض (لحد ما ندرب stage2)
# كل ما تبعد عن Urban وكل ما ورا جبل الاشارة تضعف
from scipy.ndimage import distance_transform_edt
urban_mask = (labels==0).astype(np.uint8)
dist = distance_transform_edt(1-urban_mask)
signal = 255 - np.clip(dist/5, 0, 200).astype(np.uint8)
signal[labels==1] = 20 # مية = ضعيف
signal[is_mountain] = 10 # ورا جبل = مقطوع

# 16 مستوى اشارة
signal_16 = (signal / 16).astype(np.uint8)
signal_color = (signal_16 * 16).astype(np.uint8)
Image.fromarray(signal_color).save('results/stage2_signal.png')

print("Fixed! Created colored maps:")
print("- results/stage1_landcover.png (6 colors)")
print("- results/stage2_signal.png (signal strength)")
