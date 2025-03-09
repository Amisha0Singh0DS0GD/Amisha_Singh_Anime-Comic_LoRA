import os
import random
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from config import DATASET_DIR, AUGMENTED_DIR  # Import paths from config.py

input_dir = os.path.join(DATASET_DIR, "images")
label_dir = os.path.join(DATASET_DIR, "labels")
output_dir = os.path.join(AUGMENTED_DIR, "images")
output_label_dir = os.path.join(AUGMENTED_DIR, "labels")

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Define augmentation transformations
augmentations = [
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(kernel_size=3)
]

# Get image filenames
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

if len(image_files) == 0 or len(label_files) == 0:
    raise ValueError("No images or labels found. Check dataset paths.")

print(f"Found {len(image_files)} images and {len(label_files)} labels.")

# Process images
for img_file, label_file in tqdm(zip(image_files, label_files), total=len(image_files), desc="Augmenting Data"):
    img_path = os.path.join(input_dir, img_file)
    label_path = os.path.join(label_dir, label_file)

    if not os.path.exists(label_path):
        print(f"⚠️ Skipping {img_file} (Label missing: {label_file})")
        continue

    img = Image.open(img_path).convert("RGB")
    label = Image.open(label_path).convert("RGB")

    # Save original image & label
    img.save(os.path.join(output_dir, img_file))
    label.save(os.path.join(output_label_dir, label_file))

    # Apply augmentations
    for i, aug in enumerate(augmentations):
        augmented_img = aug(img)
        augmented_label = aug(label)

        aug_img_filename = f"{img_file.split('.')[0]}_aug_{i}.png"
        aug_label_filename = f"{label_file.split('.')[0]}_aug_{i}.png"

        augmented_img.save(os.path.join(output_dir, aug_img_filename))
        augmented_label.save(os.path.join(output_label_dir, aug_label_filename))

print(f"✅ Augmentation complete! Augmented images saved in '{output_dir}'")
