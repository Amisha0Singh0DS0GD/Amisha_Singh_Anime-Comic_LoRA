import torch
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import PeftModel
import os

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

# Load Stable Diffusion Pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
).to(device)

# Load fine-tuned LoRA model
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device, dtype=dtype)
unet = PeftModel.from_pretrained(unet, "fine-tuned-unet-lora").to(device, dtype=dtype)
pipe.unet = unet

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Function to generate anime-style image
def generate_anime_image(image_path, output_path="output.png"):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device, dtype=dtype)

    with torch.no_grad():
        generated_image = pipe(prompt="", image=image).images[0]  # Inference using Stable Diffusion

    generated_image.save(output_path)
    print(f"Anime-style image saved to {output_path}")

# Example Usage
if __name__ == "__main__":
    test_image_path = "test.jpg"  # Change to your input image path
    generate_anime_image(test_image_path, "anime_output.png")
