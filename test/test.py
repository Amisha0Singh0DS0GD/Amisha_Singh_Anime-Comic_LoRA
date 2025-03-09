import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import PeftModel
from torchvision import transforms
from PIL import Image
import os

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
print(f"Using device: {device}")

# Load Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
).to(device)

# Load fine-tuned UNet model
unet = UNet2DConditionModel.from_pretrained("fine-tuned-unet-lora").to(device, dtype=dtype)
pipe.unet = unet

print("Model loaded successfully.")

# Define transform for input image
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load and preprocess real image
input_image_path = "test_images/sample_real.jpg"
if not os.path.exists(input_image_path):
    raise FileNotFoundError(f"Test image not found at {input_image_path}")

real_image = Image.open(input_image_path).convert("RGB")
real_image = transform(real_image).unsqueeze(0).to(device, dtype=dtype)

# Generate anime-stylized image
with torch.no_grad():
    generated_image = pipe(prompt="anime-style transformation", image=real_image).images[0]

# Save the output
output_path = "output/anime_result.jpg"
generated_image.save(output_path)
print(f"Anime-stylized image saved at {output_path}")
