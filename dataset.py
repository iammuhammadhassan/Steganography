import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------
# 1. THE SRM HIGH-PASS FILTER (The X-Ray for Steganography)
# ---------------------------------------------------------
class SRMConv2d(nn.Module):
    """
    Applies Spatial Rich Model (SRM) high-pass filters to images.
    This destroys the image content and extracts the microscopic noise residuals.
    """
    def __init__(self):
        super(SRMConv2d, self).__init__()
        
        # Three standard SRM filters used in Steganalysis
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
                   
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
                   
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
                   
        # Normalize the filters
        filter1 = np.array(filter1) / 4.0
        filter2 = np.array(filter2) / 12.0
        filter3 = np.array(filter3) / 2.0
        
        # Stack them into a PyTorch tensor (3 filters, 1 input channel, 5x5 size)
        filters = np.stack([filter1, filter2, filter3], axis=0)
        filters = torch.FloatTensor(filters).unsqueeze(1)
        
        # Set them as fixed weights (the AI does not train these, they are static filters)
        self.weight = nn.Parameter(filters, requires_grad=False)
        
    def forward(self, x):
        # Apply the filters to the incoming image batch
        return nn.functional.conv2d(x, self.weight, stride=1, padding=2)

# ---------------------------------------------------------
# 2. THE PYTORCH DATASET LOADER
# ---------------------------------------------------------
class StegoDataset(Dataset):
    def __init__(self, cover_dir, lsb_dir, max_samples_per_class=2500):
        self.filepaths = []
        self.labels = []
        
        # Load Cover Images (Class 0)
        cover_images = [os.path.join(cover_dir, f) for f in os.listdir(cover_dir) if f.endswith(('.png', '.jpg'))]
        cover_images = cover_images[:max_samples_per_class] # Force balance
        self.filepaths.extend(cover_images)
        self.labels.extend([0] * len(cover_images))
        
        # Load LSB Stego Images (Class 1)
        lsb_images = [os.path.join(lsb_dir, f) for f in os.listdir(lsb_dir) if f.endswith('.png')]
        lsb_images = lsb_images[:max_samples_per_class] # Force balance
        self.filepaths.extend(lsb_images)
        self.labels.extend([1] * len(lsb_images))
        
        # Transform: Convert to Grayscale -> Resize to 256x256 -> Convert to Tensor
        # We use Grayscale because noise residuals are best detected on a single luminance channel
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        # Initialize the SRM filter
        self.srm_filter = SRMConv2d()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]
        
        # Open and transform image
        img = Image.open(img_path)
        img_tensor = self.transform(img) # Shape: (1, 256, 256)
        
        # Apply SRM Filter to extract the noise map
        # We add a batch dimension (1, 1, 256, 256) for the Conv2d, then remove it
        noise_map = self.srm_filter(img_tensor.unsqueeze(0)).squeeze(0) # Shape: (3, 256, 256)
        
        return noise_map, label

# ---------------------------------------------------------
# TEST BLOCK (To make sure it works)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Testing the Data Loader and SRM Filters...")
    dataset = StegoDataset(cover_dir='data/class_0_cover', lsb_dir='data/class_1_lsb')
    print(f"Total balanced dataset size: {len(dataset)} images")
    
    # Grab the first image to see what the data looks like
    noise_map, label = dataset[0]
    print(f"Noise Map Tensor Shape: {noise_map.shape} (3 filters, 256 height, 256 width)")
    print(f"Label: {label} (0 = Cover, 1 = LSB)")