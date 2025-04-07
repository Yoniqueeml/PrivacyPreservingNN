import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def encrypt(image, key=42):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy() * 255.0  # (H, W, C), [0, 255]

    h, w, _ = image.shape
    image = image.astype(np.float32) / 255.0  # [0, 1]

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

    encrypted = np.zeros_like(image)
    for c in range(3):
        encrypted[:, :, c] = cv2.remap(
            image[:, :, c],
            x_new, y_new,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

    # 2. Гладкое преобразование цветов
    a = np.random.randn(3, 3)
    encrypted = np.dot(encrypted, a)
    encrypted = np.sin(encrypted * np.pi)

    encrypted = (encrypted - encrypted.min()) / (encrypted.max() - encrypted.min())
    encrypted = (encrypted * 255).astype(np.uint8)

    # Конвертируем обратно в тензор (C, H, W)
    return transforms.ToTensor()(encrypted)


# Пример использования
img = Image.open(r"D:\University\Course5\NIRS\MNIST\Test\filters\images\000021.jpg").convert("RGB")
transform = transforms.ToTensor()
image_tensor = transform(img).unsqueeze(0)  # (1, C, H, W)

# Шифруем изображение
encrypted_tensor = encrypt(image_tensor[0])  # (C, H, W)


# Визуализация
def imshow(tensor):
    plt.imshow(tensor.permute(1, 2, 0))  # (H, W, C)
    plt.axis('off')
    plt.show()


imshow(image_tensor[0])  # Исходное
imshow(encrypted_tensor)  # Зашифрованное