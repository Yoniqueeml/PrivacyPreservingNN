import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def encrypt(image, key=42):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy() * 255.0

    if image.ndim == 2:
        image = image[..., np.newaxis]

    h, w = image.shape[:2]
    image = image.astype(np.float32) / 255.0

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


image_path = rf'D:\University\Course5\NIRS\MNIST\Test\filters\images\000005.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Загружаем в grayscale

if original_image is None:
    raise FileNotFoundError(f"Изображение {image_path} не найдено")

encrypted_image = encrypt(original_image)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Оригинальное изображение')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(encrypted_image.squeeze(), cmap='gray')
plt.title('Зашифрованное изображение')
plt.axis('off')

plt.tight_layout()
plt.show()