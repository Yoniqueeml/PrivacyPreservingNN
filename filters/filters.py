import os

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter

def encrypt(image, key=42):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy() * 255.0

    if image.ndim == 2:
        image = image[..., np.newaxis]

    h, w = image.shape[:2]
    image = image.astype(np.float32) / 255.0

    image = (image ** 5) + (0.3 * np.sin(image * 6 * np.pi)) + (0.3 * np.cos(image * 6 * np.pi))

    np.random.seed(key)
    freq_x = np.random.uniform(0.05, 0.2)
    freq_y = np.random.uniform(0.05, 0.2)
    phase_x = np.random.uniform(0, 2 * np.pi)
    phase_y = np.random.uniform(0, 2 * np.pi)

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    x_new = x + 10 * np.sin(freq_x * x + phase_x)
    y_new = y + 10 * np.sin(freq_y * y + phase_y)

    x_new = np.clip(x_new, 0, w - 1)
    y_new = np.clip(y_new, 0, h - 1)

    encrypted = np.zeros_like(image)
    for c in range(image.shape[2]):
        encrypted[:, :, c] = cv2.remap(
            image[:, :, c],
            x_new.astype(np.float32), y_new.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

    if encrypted.shape[2] == 3:
        a = np.random.randn(3, 3)
        encrypted = np.dot(encrypted, a)
        encrypted = np.sin(encrypted * np.pi)

    encrypted = (encrypted - encrypted.min()) / (encrypted.max() - encrypted.min())
    encrypted = (encrypted * 255).astype(np.uint8)

    return encrypted


def apply_mask(data, masked_percent):
    if isinstance(data, np.ndarray):
        mask = (np.random.rand(*data.shape) > masked_percent / 100).astype(np.float32)
    elif isinstance(data, torch.Tensor):
        mask = (torch.rand(data.size(), device=data.device) > masked_percent / 100).float()
    else:
        raise ValueError("data should be a numpy array or torch tensor.")

    return (data * mask).astype(int)

def apply_denoising_filters(data):
    median_filtered = median_filter(data, size=3)
    gaussian_filtered = gaussian_filter(data, sigma=1)
    return median_filtered, gaussian_filtered

def load_images_from_folder(folder, num_images=3):
    images = []
    for idx, filename in enumerate(os.listdir(folder)):
        if idx >= num_images:
            break
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path)
            img = np.array(img) / 255.0
            images.append(img)
    return images

folder_path = rf'images'
images = load_images_from_folder(folder_path)

for img_idx, img in enumerate(images):
    encrypted_data = encrypt(img)
    masked_data = apply_mask(encrypted_data, 0)
    median_filtered, gaussian_filtered = apply_denoising_filters(masked_data)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f"Original Image {img_idx + 1}")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(encrypted_data)
    axes[0, 1].set_title(f"Encrypted Image {img_idx + 1}")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(masked_data)
    axes[0, 2].set_title(f"Masked Image {img_idx + 1} (30%)")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(median_filtered)
    axes[1, 0].set_title(f"Median Filtered Image {img_idx + 1}")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gaussian_filtered)
    axes[1, 1].set_title(f"Gaussian Filtered Image {img_idx + 1}")
    axes[1, 1].axis('off')

    plt.show()