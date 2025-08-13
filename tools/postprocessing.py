from google.colab import drive
drive.mount('/content/drive')

# Download and Extract Data
def extract_brats_data(zip_path, extract_to):
    """Extract BraTS data from zip file"""
    print(f"Extracting data from {zip_path}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"Data extracted to {extract_to}")

import os

search_dir = '/content/drive/My Drive'
zip_files = []

for root, dirs, files in os.walk(search_dir):
    for file in files:
        if file.endswith(".zip"):
            zip_files.append(os.path.join(root, file))

zip_files  # This will list all found ZIP files and their full paths

# Extract Training Data
from google.colab import drive
import zipfile
import os

# Mount Google Drive

drive.mount('/content/drive')


# Define Paths

zip_path = '/content/drive/MyDrive/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData.zip'  # Update if needed
extract_dir = '/content/drive/MyDrive/BraTS2023_TrainingData'  # Destination folder

# Create extract folder if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)


#  Extract ZIP File

print(f"Extracting {zip_path} to {extract_dir} ...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(" Extraction completed!")
print(f"Extracted files are in: {extract_dir}")


# Verify Files

print("Sample files after extraction:")
for root, dirs, files in os.walk(extract_dir):
    for file in files[:10]:  # Show first 10 files
        print(os.path.join(root, file))
    break

# Extracing validation Data
import zipfile
import os


#  Mount Google Drive

drive.mount('/content/drive')


#  Define Paths

zip_path = '/content/drive/My Drive/Baba/BraTS2024-SSA-Challenge-ValidationData.zip'
extract_dir = '/content/drive/MyDrive/BraTS2024_ValidationData'  # Destination folder

# Create extract folder if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# ===============================
# Extract ZIP File
# ===============================
print(f"Extracting {zip_path} to {extract_dir} ...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(" Extraction completed!")
print(f"Extracted files are in: {extract_dir}")

# ===============================
# Verify Files
# ===============================
print("Sample files after extraction:")
for root, dirs, files in os.walk(extract_dir):
    for file in files[:10]:  # Show first 10 files
        print(os.path.join(root, file))
    break
