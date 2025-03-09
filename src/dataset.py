# dataset.py
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class AnimeStyleTransferDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.image_filenames[idx])

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label