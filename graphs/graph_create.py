import os
import matplotlib.pyplot as plt
import re

# Папка с данными
data_folder = 'data'

# Словарь для хранения данных
losses = {}
accuracies = {}

# Пройдите по папкам внутри data_folder
for dataset_name in ['breast', 'mnist']:
    dataset_folder = os.path.join(data_folder, dataset_name)

    # Пройдите по файлам в папке
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".txt"):
            # Извлеките процент маскирования из имени файла
            match = re.search(r'(\d+)%', filename)
            if match:
                masked_percent = int(match.group(1))
                key = f'{dataset_name}_{filename[:-4]}'  # Используйте полное имя файла без расширения
                losses[key] = {'epochs': [], 'losses': []}
                accuracies[key] = {'final_accuracy': None}

                # Считайте файл
                with open(os.path.join(dataset_folder, filename), 'r') as f:
                    lines = f.readlines()
                    for line in lines[1:-1]:  # Пропустите заголовок и последнюю строку
                        values = line.strip().split('\t')
                        if len(values) >= 2:  # Проверьте, что строка содержит хотя бы два значения
                            epoch = int(values[0])
                            loss = float(values[1])
                            losses[key]['epochs'].append(epoch)
                            losses[key]['losses'].append(loss)

                    # Прочитайте последнюю строку и извлеките значение точности
                    last_line = lines[-1].strip()
                    match = re.search(r'Final Accuracy: (\d+\.\d+)%', last_line)
                    if match:
                        final_accuracy = float(match.group(1)) / 100
                        accuracies[key]['final_accuracy'] = final_accuracy

# Данные MobileNetV2
mobilenet_data = {
    '0_masked': {
        'epochs': list(range(1, 11)),
        'train_losses': [0.1733, 0.0980, 0.0744, 0.0591, 0.0434, 0.0384, 0.0338, 0.0303, 0.0307, 0.0214],
        'val_losses': [0.1158, 0.1074, 0.1187, 0.1237, 0.1323, 0.1352, 0.1361, 0.1450, 0.1325, 0.1397],
        'train_accuracies': [0.9408, 0.9645, 0.9731, 0.9781, 0.9844, 0.9867, 0.9881, 0.9897, 0.9892, 0.9929],
        'val_accuracies': [0.9585, 0.9621, 0.9592, 0.9584, 0.9555, 0.9567, 0.9588, 0.9578, 0.9584, 0.9597]
    },
    '30_masked': {
        'epochs': list(range(1, 11)),
        'train_losses': [0.2232, 0.1617, 0.1395, 0.1218, 0.1060, 0.0903, 0.0803, 0.0674, 0.0613, 0.0529],
        'val_losses': [0.2017, 0.2100, 0.1925, 0.2338, 0.1954, 0.2383, 0.2072, 0.2270, 0.2373, 0.2440],
        'train_accuracies': [0.9260, 0.9410, 0.9482, 0.9543, 0.9601, 0.9658, 0.9698, 0.9752, 0.9769, 0.9805],
        'val_accuracies': [0.9339, 0.9334, 0.9341, 0.9271, 0.9364, 0.9360, 0.9378, 0.9340, 0.9395, 0.9376]
    },
    'another_encrypt': {
        'epochs': list(range(1, 6)),
        'train_losses': [0.2849, 0.2363, 0.2294, 0.2218, 0.2155],
        'val_losses': [0.2610, 0.2459, 0.2424, 0.2358, 0.2944],
        'train_accuracies': [0.9041, 0.9208, 0.9203, 0.9220, 0.9237],
        'val_accuracies': [0.9222, 0.9213, 0.9218, 0.9213, 0.9086]
    }
}

# Постройте график
plt.figure(figsize=(15, 5))

# График для breast
plt.subplot(1, 3, 1)
for key, data in losses.items():
    if 'breast' in key:
        plt.plot(data['epochs'], data['losses'], label=f'{key} Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Breast Loss')
plt.legend()
plt.yscale('log')

# График для mnist
plt.subplot(1, 3, 2)
for key, data in losses.items():
    if 'mnist' in key:
        plt.plot(data['epochs'], data['losses'], label=f'{key} Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MNIST Loss')
plt.legend()
plt.yscale('log')

# График для MobileNetV2
plt.subplot(1, 3, 3)
for key, data in mobilenet_data.items():
    plt.plot(data['epochs'], data['train_losses'], label=f'{key} Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MobileNetV2 Loss')
plt.legend()
plt.yscale('log')

plt.tight_layout()
plt.show()

# Выведите конечные точности
print("Конечные точности:")
for key, data in accuracies.items():
    if data['final_accuracy'] is not None:
        print(f"{key}: {data['final_accuracy'] * 100}%")

for key, data in mobilenet_data.items():
    print(f"MobileNetV2 {key} Validation Accuracy: {data['val_accuracies'][-1] * 100}%")
