import torch
import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim # type: ignore
from skimage.metrics import peak_signal_noise_ratio as psnr # type: ignore
from torch.nn.functional import mse_loss

# Function to load and preprocess images
def load_image(image_path, size=(512, 512)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    return image

# Function to calculate evaluation metrics
def evaluate_images(real_images_dir, generated_images_dir):
    real_images = sorted(os.listdir(real_images_dir))
    generated_images = sorted(os.listdir(generated_images_dir))

    assert len(real_images) == len(generated_images), "Mismatch in number of images."

    ssim_scores = []
    psnr_scores = []
    mse_scores = []

    for real_img_name, gen_img_name in zip(real_images, generated_images):
        real_img_path = os.path.join(real_images_dir, real_img_name)
        gen_img_path = os.path.join(generated_images_dir, gen_img_name)

        real_img = load_image(real_img_path)
        gen_img = load_image(gen_img_path)

        # Compute SSIM
        ssim_value = ssim(real_img, gen_img, multichannel=True, data_range=1.0)
        ssim_scores.append(ssim_value)

        # Compute PSNR
        psnr_value = psnr(real_img, gen_img, data_range=1.0)
        psnr_scores.append(psnr_value)

        # Compute MSE
        real_tensor = torch.tensor(real_img).permute(2, 0, 1).unsqueeze(0)  # (C, H, W) format
        gen_tensor = torch.tensor(gen_img).permute(2, 0, 1).unsqueeze(0)
        mse_value = mse_loss(real_tensor, gen_tensor).item()
        mse_scores.append(mse_value)

    # Print average metrics
    print("\nEvaluation Metrics:")
    print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
    print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
    print(f"Average MSE: {np.mean(mse_scores):.6f}")

    return {
        "ssim": np.mean(ssim_scores),
        "psnr": np.mean(psnr_scores),
        "mse": np.mean(mse_scores)
    }

if __name__ == "__main__":
    real_images_dir = "dataset_augmented/test/labels"   # Path to ground truth anime images
    generated_images_dir = "output/generated_anime"     # Path to model-generated images

    results = evaluate_images(real_images_dir, generated_images_dir)
