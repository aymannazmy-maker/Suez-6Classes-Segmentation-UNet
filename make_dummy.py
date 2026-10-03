import numpy as np, rasterio, os
os.makedirs('data/images', exist_ok=True)
os.makedirs('data/masks', exist_ok=True)
for i in range(4):
    img = (np.random.rand(4,256,256)*10000).astype('uint16')
    with rasterio.open(f'data/images/img_{i}.tif','w',driver='GTiff',height=256,width=256,count=4,dtype='uint16') as dst:
        dst.write(img)
    mask = (np.random.randint(0,6,(256,256))).astype('uint8')
    with rasterio.open(f'data/masks/mask_{i}.tif','w',driver='GTiff',height=256,width=256,count=1,dtype='uint8') as dst:
        dst.write(mask,1)
print('Dummy data created OK')
