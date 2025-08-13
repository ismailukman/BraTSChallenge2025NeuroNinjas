# Preprocessing Code Training Data( Subset of BraTS Dataset to 2D for MedSAM)
import os
import nibabel as nib
import numpy as np
import cv2
import json
from tqdm import tqdm

# === Paths ===
data_dir = "/content/drive/MyDrive/BraTS2023_TrainingData/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
output_dir = "/content/drive/MyDrive/BraTS2023_processed"
modalities = ["t1n", "t1c", "t2w", "t2f"]

# Create output folders
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "bboxes"), exist_ok=True)

# Helper Functions
def get_bounding_box(mask):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def normalize_image(img):
    img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

# Main Loop
for subject in tqdm(os.listdir(data_dir)):
    subj_path = os.path.join(data_dir, subject)
    if not os.path.isdir(subj_path):
        continue

    try:
        # Find all files in this subject folder
        all_files = os.listdir(subj_path)

        imgs = []
        for mod in modalities:
            mod_file = [f for f in all_files if f.endswith(f"{mod}.nii.gz")]
            if not mod_file:
                print(f" Missing {mod} for {subject}, skipping.")
                break
            nii = nib.load(os.path.join(subj_path, mod_file[0]))
            imgs.append(normalize_image(nii.get_fdata()))

        if len(imgs) < 4:
            continue

        seg_file = [f for f in all_files if f.endswith("seg.nii.gz")]
        if not seg_file:
            print(f" Missing segmentation for {subject}, skipping.")
            continue

        seg = nib.load(os.path.join(subj_path, seg_file[0])).get_fdata()

        for i in range(imgs[0].shape[2]):  # axial slices
            slice_mods = [m[:, :, i] for m in imgs]
            slice_mask = (seg[:, :, i] > 0).astype(np.uint8)

            if np.sum(slice_mask) == 0:
                continue

            # Resize to 256x256
            image = np.stack(slice_mods, axis=-1)
            image_resized = cv2.resize(image, (256, 256))
            mask_resized = cv2.resize(slice_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

            bbox = get_bounding_box(mask_resized)
            if bbox is None:
                continue

            base_name = f"{subject}_slice_{i}"
            np.save(os.path.join(output_dir, "images", f"{base_name}.npy"), image_resized)
            np.save(os.path.join(output_dir, "masks", f"{base_name}.npy"), mask_resized)
            with open(os.path.join(output_dir, "bboxes", f"{base_name}.json"), "w") as f:
                json.dump({"bbox": bbox}, f)

    except Exception as e:
        print(f" Failed to process {subject}: {e}")

# Preprocessing Code validation Data( Subset of BraTS Dataset to 2D for MedSAM)
import os
import nibabel as nib
import numpy as np
import cv2
import json
from tqdm import tqdm

# === Paths ===
data_dir = "/content/drive/MyDrive/BraTS2024_ValidationData/BraTS2024-SSA-Challenge-ValidationData"
output_dir = "/content/drive/MyDrive/BraTS2024_validation_processed"
modalities = ["t1n", "t1c", "t2w", "t2f"]

# Create output folders
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "bboxes"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
def get_bounding_box(mask):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def normalize_image(img):
    img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

# Preprocessing Loop
for subject in tqdm(os.listdir(data_dir)):
    subj_path = os.path.join(data_dir, subject)

    if not os.path.isdir(subj_path):
        continue
    if not subject.startswith("BraTS"):
        continue

    try:
        imgs = []
        for mod in modalities:
            mod_file = os.path.join(subj_path, f"{subject}-{mod}.nii.gz")
            if not os.path.exists(mod_file):
                print(f" Missing {mod} for {subject}, skipping.")
                imgs = []
                break
            nii = nib.load(mod_file)
            imgs.append(normalize_image(nii.get_fdata()))

        if len(imgs) < 4:
            continue  # Skip incomplete cases

        for i in range(imgs[0].shape[2]):  # Slices
            slice_mods = [m[:, :, i] for m in imgs]
            image = np.stack(slice_mods, axis=-1)  # (H, W, 4)
            image_resized = cv2.resize(image, (256, 256))

            base_name = f"{subject}_slice_{i}"
            np.save(os.path.join(output_dir, "images", f"{base_name}.npy"), image_resized)

    except Exception as e:
        print(f" Failed to process {subject}: {e}")

print(f" Preprocessing complete! Saved in {output_dir}")