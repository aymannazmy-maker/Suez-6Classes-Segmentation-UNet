import torch
from torch.utils.data import DataLoader
from dataset_real import SuezDataset
from model_unet import UNet
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

dataset = SuezDataset()
if len(dataset)==0:
    print("!! لسه محملتش الداتا في فولدر data/images و data/masks")
    print("اعمل فولدر data وحط الصور جواه")
    exit()

loader = DataLoader(dataset, batch_size=2, shuffle=True)
model = UNet(n_classes=6, in_channels=4).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    total_loss=0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        # resize for UNet if needed
        if img.shape[-1]!=256:
            img = torch.nn.functional.interpolate(img, size=(256,256))
            mask = torch.nn.functional.interpolate(mask.unsqueeze(1).float(), size=(256,256), mode='nearest').squeeze(1).long()
        pred = model(img)
        loss = criterion(pred, mask)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss+=loss.item()
    print(f"Epoch {epoch+1}/50 - Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), f"unet_suez_6classes_epoch{epoch+1}.pth")

print("Training Done!")
