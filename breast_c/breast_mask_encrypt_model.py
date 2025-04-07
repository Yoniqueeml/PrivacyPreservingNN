import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv(r'data\data_breast.csv')
print(f'isnull.sum: {df.isnull().sum()}')

df.drop(columns=['Unnamed: 32', 'id'], inplace=True, errors='ignore')

label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])
features = df.drop(columns=['diagnosis']).values
labels = df['diagnosis'].values

scaler = MinMaxScaler()
features = scaler.fit_transform(features)


def encrypt(data):
    encrypted_data = (data ** 5) + (0.3 * np.sin(data * 6 * np.pi)) + (0.3 * np.cos(data * 6 * np.pi))
    return encrypted_data

encrypted_features = encrypt(features)

X_train, X_test, y_train, y_test = train_test_split(encrypted_features, labels, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


class DiseaseClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DiseaseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = 2
learning_rate = 0.001
num_epochs = 10
batch_size = 64

model = DiseaseClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

masked_percent = 0

def apply_mask(data, masked_percent):
    mask = (torch.rand(data.size()) > masked_percent / 100).float()
    return data * mask

def calculate_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y).sum().item()
    return correct / y.size(0)


epoch_losses = []
epoch_accuracies = []

X_test = apply_mask(X_test, masked_percent)

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        masked_inputs = apply_mask(inputs, masked_percent)
        outputs = model(masked_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(epoch_loss)

    accuracy = calculate_accuracy(model, X_test, y_test)
    epoch_accuracies.append(accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {accuracy * 100:.2f}%")

# Сохраните результаты в файл
with open(f"{masked_percent}%_masked_encrypted_results.txt", "w") as f:
    f.write("Epoch\tLoss\tAccuracy\n")
    for i in range(num_epochs):
        f.write(f"{i + 1}\t{epoch_losses[i]}\t{epoch_accuracies[i] * 100:.2f}\n")
    f.write(f"\nFinal Accuracy: {epoch_accuracies[-1] * 100:.2f}%\n")

print(f"Результаты сохранены в файл {masked_percent}%_masked_encrypted_results.txt")
