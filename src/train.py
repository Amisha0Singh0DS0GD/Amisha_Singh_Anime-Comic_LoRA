import diffusers
import torch
from torchvision import transforms
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from dataset import AnimeStyleTransferDataset
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader
import multiprocessing
import os

# Fix multiprocessing on Mac
multiprocessing.set_start_method("spawn", force=True)

# Ensure Dependencies Are Up-to-Date
print(f"Torch Version: {torch.__version__}")
print(f"Diffusers Version: {diffusers.__version__}")

# Set Device (Use CPU if MPS is buggy)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
print(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Load dataset
train_dataset = AnimeStyleTransferDataset(
    image_dir="dataset_augmented/train/images",
    label_dir="dataset_augmented/train/labels",
    transform=transform
)

# DataLoader (Reduce batch size to prevent OOM)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Load Stable Diffusion Pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
).to(device)

# Extract UNet & VAE
unet: UNet2DConditionModel = pipe.unet.to(device, dtype=dtype)
vae = pipe.vae.to(device, dtype=dtype)

# Apply LoRA using Hugging Face's PEFT
lora_config = LoraConfig(
    r=2,  # LoRA Rank
    lora_alpha=8,  # LoRA Scaling
    target_modules=["to_q", "to_k", "to_v"],  # Apply LoRA to attention layers
    lora_dropout=0.1,
    bias="none"
)

unet = get_peft_model(unet, lora_config)  # Automatically configures LoRA

# Define Optimizer (Only Train LoRA Layers)
optimizer = AdamW(unet.parameters(), lr=5e-6, weight_decay=1e-4)

# Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path="checkpoint.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

# Function to load checkpoint
def load_checkpoint(model, optimizer, checkpoint_path="checkpoint.pth"):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch
    return 0  # Start from scratch if no checkpoint

# Load checkpoint before starting training
start_epoch = load_checkpoint(unet, optimizer)

# Training Loop
num_epochs = 10
for epoch in range(start_epoch, num_epochs):
    total_loss = 0
    num_batches = len(train_loader)

    with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
        for batch_idx, (real_images, anime_images) in enumerate(train_loader):
            real_images = real_images.to(device, dtype=dtype)
            anime_images = anime_images.to(device, dtype=dtype)

            # Encode images to latent space
            with torch.no_grad():
                latent_real = vae.encode(real_images).latent_dist.mean.to(dtype=dtype) * 0.1
                latent_real = latent_real.to(device)

            # Prepare inputs for UNet
            timestep = torch.tensor([0], device=device, dtype=dtype)
            encoder_hidden_states = torch.zeros((real_images.shape[0], 77, 768), device=device, dtype=dtype)

            # Forward Pass
            outputs = None  
            try:
                outputs = unet(latent_real, timestep, encoder_hidden_states)
                if isinstance(outputs, dict):
                    outputs = outputs["sample"]
            except Exception as e:
                print(f"Forward Pass Error: {e}")
                continue  # Skip batch if error occurs

            # Convert anime images to latent space
            with torch.no_grad():
                latent_anime = vae.encode(anime_images).latent_dist.mean.to(dtype=dtype) * 0.1
                latent_anime = latent_anime.to(device)

            # Compute Loss
            if outputs is not None:
                loss = torch.nn.SmoothL1Loss()(outputs, latent_anime)

                # Backward Pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Update the progress bar
            pbar.set_postfix(loss=f"{loss.item():.6f}")
            pbar.update(1)

        # Save checkpoint after each epoch
        save_checkpoint(unet, optimizer, epoch, total_loss / num_batches if num_batches > 0 else 0)

    # Print the average loss per epoch
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Batch Loss: {avg_loss:.6f}")

# Save Fine-Tuned LoRA Weights
unet.save_pretrained("fine-tuned-unet-lora")

print("Training completed successfully!")
