# Dataset to Add Bounding Box Mask as a Prompt


from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, bbox_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.bbox_dir = bbox_dir
        self.transform = transform
        self.samples = [f.replace(".npy", "") for f in os.listdir(image_dir) if f.endswith(".npy")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]

        image = np.load(os.path.join(self.image_dir, f"{sample_id}.npy"))  # (256, 256, 4)
        mask = np.load(os.path.join(self.mask_dir, f"{sample_id}.npy"))    # (256, 256)
        with open(os.path.join(self.bbox_dir, f"{sample_id}.json"), 'r') as f:
            bbox = json.load(f)["bbox"]  # [x1, y1, x2, y2]

        # Create bounding box mask
        bbox_mask = np.zeros((256, 256), dtype=np.float32)
        x1, y1, x2, y2 = bbox
        bbox_mask[y1:y2+1, x1:x2+1] = 1.0

        # Stack image and bbox as 5 channels
        image = np.concatenate([image, bbox_mask[..., None]], axis=-1)  # shape: (256, 256, 5)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (5, 256, 256)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)        # (1, 256, 256)

        return image, mask
    # UTILITY FUNCTIONS


def normalize_image(img):
    """Normalize each modality to 0-1."""
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def load_patient_modalities(patient_path):
    """Load T1, T1ce, T2, FLAIR and create bbox mask."""
    try:
        t1 = nib.load(os.path.join(patient_path, f"{os.path.basename(patient_path)}_t1.nii.gz")).get_fdata()
        t1ce = nib.load(os.path.join(patient_path, f"{os.path.basename(patient_path)}_t1ce.nii.gz")).get_fdata()
        t2 = nib.load(os.path.join(patient_path, f"{os.path.basename(patient_path)}_t2.nii.gz")).get_fdata()
        flair = nib.load(os.path.join(patient_path, f"{os.path.basename(patient_path)}_flair.nii.gz")).get_fdata()
    except FileNotFoundError:
        return None

    # Normalize each modality
    t1, t1ce, t2, flair = [normalize_image(x) for x in [t1, t1ce, t2, flair]]

    # Dummy bounding box mask (for now, zeros)
    bbox_mask = np.zeros_like(t1, dtype=np.float32)

    return np.stack([t1, t1ce, t2, flair, bbox_mask], axis=0)  # shape: (5, H, W, D)