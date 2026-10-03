import torch
import torch.nn as nn

class PropagationUNet(nn.Module):
    """ Stage 2: 10 channels [RGB(3)+LandcoverOneHot(6)+DEM(1)] -> Signal(16 colors) """
    def __init__(self, in_channels=10, out_channels=16):
        super().__init__()
        # ده الموديل اللي هيفهم فيزياء الإشارة
        self.first_conv = nn.Conv2d(in_channels, 32, 3, padding=1)

        self.enc1 = nn.Sequential(nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU())

        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(256,128,2,2)
        self.dec2 = nn.Sequential(nn.Conv2d(256,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(128,64,2,2)
        self.dec1 = nn.Sequential(nn.Conv2d(128,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU())

        # 16 لون = 16 مستوى إشارة من -120dBm لـ -40dBm
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # x shape: [B,10,H,W]
        x0 = self.first_conv(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        y = self.up2(x3)
        y = torch.cat([y, x2], dim=1)
        y = self.dec2(y)

        y = self.up1(y)
        y = torch.cat([y, x1], dim=1)
        y = self.dec1(y)

        return self.final(y) # [B,16,H,W] signal map

# شرح الفيزياء للموديل
PHYSICS_GUIDE = """
الموديل هيفهم ان:
- Channel 0 (Urban): مبنى = انعكاس + تشتيت -> Signal يضعف 10-20 dB
- Channel 1 (Water): مية = امتصاص عالي -> Signal يضعف 30 dB
- Channel 2 (Veg): شجر = تشتيت خفيف
- Channel 4 (Mountain): جبل = قطع كامل للإشارة وراه Shadow
- Channel 9 (DEM): كل ما الارتفاع يزيد، الإشارة تتقطع
"""
