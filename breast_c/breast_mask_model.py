import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

df = pd.read_csv(rf'data\data_breast.csv')
print(f'isnull.sum: {df.isnull().sum()}')

df.drop(columns=['Unnamed: 32', 'id'], inplace=True, errors='ignore')

label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])
features = df.drop(columns=['diagnosis']).values
labels = df['diagnosis'].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

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
num_epochs = 20
batch_size = 16

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
    accuracy = calculate_accuracy(model, X_test, y_test)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy * 100:.2f}%")


torch.save(model.state_dict(), f'weights_mask_{masked_percent}.pth')
print(f"Model weights saved to weights_mask_{masked_percent}.pth")