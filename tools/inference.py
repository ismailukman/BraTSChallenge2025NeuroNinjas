# Simplified MedSAM Model (from Scratch) Model to Accept 5-Channel Input
import torch.nn as nn

class SimpleMedSAM(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):  # Updated: 5 channels (4 MRI + bbox mask)
        super(SimpleMedSAM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decoder(x)
        return x

# Dice Score for Evaluation

def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

import torch
import torch.nn as nn


# Conv Block

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# UpConv Block

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


# MedSAM-like U-Net

class MedSAMUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super().__init__()

        # Encoder
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv5 = ConvBlock(512, 1024)

        # Decoder
        self.up4 = UpConv(1024, 512)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = UpConv(512, 256)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = UpConv(256, 128)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = UpConv(128, 64)
        self.dec1 = ConvBlock(128, 64)

        # Output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        bn = self.conv5(p4)

        # Decoder
        u4 = self.up4(bn)
        m4 = torch.cat([u4, c4], dim=1)
        d4 = self.dec4(m4)

        u3 = self.up3(d4)
        m3 = torch.cat([u3, c3], dim=1)
        d3 = self.dec3(m3)

        u2 = self.up2(d3)
        m2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec2(m2)

        u1 = self.up1(d2)
        m1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec1(m1)

        out = self.final(d1)
        return out

# Training Loop
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# BraTSDataset and SimpleMedSAM are already defined

# Paths
image_dir = "/content/drive/MyDrive/BraTS2023_processed/images"
mask_dir = "/content/drive/MyDrive/BraTS2023_processed/masks"
bbox_dir = "/content/drive/MyDrive/BraTS2023_processed/bboxes"

# Dataset & Loader
dataset = BraTSDataset(image_dir, mask_dir, bbox_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model, Loss, Optimizer
model = SimpleMedSAM(in_channels=5, out_channels=1).cuda()
optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

# Dice Score Function
def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Training Loop
epochs = 10
save_dir = "/content/drive/MyDrive/MedSAM"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(epochs):
    model.train()
    total_loss, total_dice = 0.0, 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
    for images, masks in progress_bar:
        images, masks = images.cuda(), masks.cuda()

        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute Dice
        dice = dice_score(torch.sigmoid(preds), masks)

        # Update accumulators
        total_loss += loss.item()
        total_dice += dice.item()

        #  Show Batch Loss and Dice in tqdm
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Dice": f"{dice:.4f}"})

    # Epoch averages
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)

    #  Print summary after each epoch
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Avg Dice: {avg_dice:.4f}")

    #  Save checkpoint once per epoch (not every batch)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

    import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class MedSAMUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super(MedSAMUNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.dec1 = DoubleConv(512 + 256, 256)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec3 = DoubleConv(128 + 64, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        d1 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, x3], dim=1)
        d1 = self.dec1(d1)

        d2 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d3 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, x1], dim=1)
        d3 = self.dec3(d3)

        return self.final(d3)
    
    # Predict → Reconstruct 3D → Zip
import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import zipfile

# ----------------- Model Definition -----------------
import torch.nn as nn

class MedSAMUNet(nn.Module):
    def __init__(self, n_channels=5):
        super(MedSAMUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        e = self.encoder(x)
        p = self.pool(e)
        out = self.decoder(p)
        return out

# ----------------- Paths -----------------
INPUT_DIR = "/content/drive/MyDrive/BraTS2024_validation_processed/images"  # CHANGE this to actual path
OUTPUT_DIR = "/content/drive/MyDrive/BraTS2025_predictions_3D"
MODEL_PATH = "/content/drive/MyDrive/MedSAM/checkpoint_epoch_10.pth"
ZIP_PATH = "/content/drive/MyDrive/BraTS2025_predictions_3D.zip"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- Load Model -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MedSAMUNet(n_channels=5).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(" Model Loaded")

# ----------------- Group slices by case -----------------
cases = {}
for f in os.listdir(INPUT_DIR):
    if f.endswith(".npy"):
        case_id = "_".join(f.split("_")[:3])  # e.g., BraTS-SSA-00157-000
        if case_id not in cases:
            cases[case_id] = []
        cases[case_id].append(f)

print(f"Found {len(cases)} cases")

affine = np.eye(4)

# ----------------- Inference & Reconstruction -----------------
for case_id, files in tqdm(cases.items()):
    files.sort(key=lambda x: int(x.split("_slice_")[-1].replace(".npy", "")))

    pred_slices = []
    for fname in files:
        img_path = os.path.join(INPUT_DIR, fname)
        image = np.load(img_path)  # Could be (H, W, 4)

        # Fix shape: (H, W, 4) → (4, H, W)
        if image.ndim == 3 and image.shape[-1] == 4:
            image = np.transpose(image, (2, 0, 1))
        elif image.shape[0] != 4:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # Normalize each channel
        for i in range(image.shape[0]):
            if image[i].std() > 0:
                image[i] = (image[i] - image[i].mean()) / image[i].std()

        # Add extra channel
        extra_channel = np.zeros_like(image[0])
        image = np.concatenate([image, extra_channel[np.newaxis, ...]], axis=0)  # (5, H, W)

        # Predict
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            mask = (torch.sigmoid(output) > 0.5).cpu().numpy()[0, 0]

        pred_slices.append(mask)

    # Stack into 3D
    volume = np.stack(pred_slices, axis=-1)  # shape: (H, W, D)

    # Save as NIfTI
    case_folder = os.path.join(OUTPUT_DIR, case_id)
    os.makedirs(case_folder, exist_ok=True)
    out_path = os.path.join(case_folder, "seg.nii.gz")
    nib.save(nib.Nifti1Image(volume.astype(np.uint8), affine), out_path)
    print(f"Saved {out_path}")

print(" All predictions saved as 3D NIfTI")

# ----------------- Zip the predictions -----------------
print("Zipping predictions...")
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, OUTPUT_DIR)
            zipf.write(file_path, arcname)
print(f" Zipped predictions: {ZIP_PATH}")


