import os
import struct

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        image_data = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return image_data


def read_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        label_data = np.fromfile(f, dtype=np.uint8)
    return label_data


def show_examples():
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(train_x[i], cmap='gray')
        plt.title(f'Label: {train_y[i]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# show_examples()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16,
                               5)  # (channels,output,kernel_size)   [Batch_size,1,28,28]  --> [Batch_size,16,24,24]
        self.mxp1 = nn.MaxPool2d(2)  # [Batch_size,16,24,24] --> [Batch_size,16,24/2,24/2] --> [Batch_size,16,12,12]
        self.conv2 = nn.Conv2d(16, 24, 5)  # [Batch_size,16,12,12] --> [Batch_size,24,8,8]
        self.mxp2 = nn.MaxPool2d(2)  # [Batch_size,24,8,8] ---> [Batch_size,32,8/2,8/2] ---> [Batch_size,24,4,4]
        self.linear1 = nn.Linear(24 * 4 * 4, 100)  # input shape --> 100 outputs
        self.linear2 = nn.Linear(100, 10)  # 100 inputs --> 10 outputs

    def forward(self, x):
        X = self.mxp1(F.relu(self.conv1(x)))
        X = self.mxp2(F.relu(self.conv2(X)))
        X = X.view(-1, 24 * 4 * 4)  # reshaping to input shape
        X = F.relu(self.linear1(X))
        X = self.linear2(X)
        return F.log_softmax(X, dim=1)


def compute_val_loss(val_x, val_y, model):
    model.eval()
    output = model(val_x)
    loss = F.cross_entropy(output, val_y)
    return loss

def apply_mask(data, masked_percent):
    mask = (torch.rand(data.size()) > masked_percent / 100).float()
    return data * mask


train_x = read_mnist_images(rf'D:\University\Course5\NIRS\MNIST\Test\mnist_real\data\train\train-images-idx3-ubyte')
train_y = read_mnist_labels(rf'D:\University\Course5\NIRS\MNIST\Test\mnist_real\data\train_labels\train-labels-idx1-ubyte')

train_x = train_x / 255.0

test_x = read_mnist_images(rf'D:\University\Course5\NIRS\MNIST\Test\mnist_real\data\test\t10k-images.idx3-ubyte')
test_y = read_mnist_labels(rf'D:\University\Course5\NIRS\MNIST\Test\mnist_real\data\test_labels\t10k-labels.idx1-ubyte')

test_x = test_x / 255.0

trn_x = torch.from_numpy(train_x).type(torch.FloatTensor).view(-1, 1, 28, 28)
trn_y = torch.from_numpy(train_y).type(torch.LongTensor)

val_x = torch.from_numpy(test_x).type(torch.FloatTensor).view(-1, 1, 28, 28)
val_y = torch.from_numpy(test_y).type(torch.LongTensor)

cnn = Model()
optimizer = Adam(cnn.parameters(), lr=1e-3)
EPOCHS = 10
masked_percent = 0  # x% = 0
weights_file_path = f'{masked_percent}%_masked.pth'
batch_size = 1024

trn_loss = []
val_loss = []

trn = TensorDataset(trn_x, trn_y)
trn = DataLoader(trn, batch_size=batch_size)

val_x = apply_mask(val_x, masked_percent)

if os.path.exists(weights_file_path) and 1 == 0:
    print("Loading weights...")
    cnn.load_state_dict(torch.load(weights_file_path))
    cnn.eval()
else:
    cnn.train()
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(trn):
            optimizer.zero_grad()

            masked_data = apply_mask(data, masked_percent)
            y_pred = cnn(masked_data)

            loss = F.cross_entropy(y_pred, target)
            trn_loss.append(loss.cpu().data.item())

            loss.backward()
            optimizer.step()
            loss = compute_val_loss(val_x, val_y, cnn)
            val_loss.append(loss.cpu().data.item())

        print("Epoch: {} | loss: {} | val_loss: {}".format(epoch + 1, trn_loss[-1], val_loss[-1]))

    #torch.save(cnn.state_dict(), f'{masked_percent}%_masked.pth')
    #print(f"Weights saved to {masked_percent}%_masked.pth")

    plt.figure(figsize=(5, 5), dpi=200)
    plt.plot(trn_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.legend(loc='upper right')


def predict_with_pytorch(model, val_x):
    model.eval()
    y_preds = []
    out = model(val_x)
    _, predicted = torch.max(out.data, 1)
    for p in predicted:
        y_preds.append(p.detach().cpu().numpy().item())

    return y_preds


pred = predict_with_pytorch(cnn, val_x)
acc = accuracy_score(val_y.numpy(), pred)

print("Accuracy:", acc * 100, "%")
with open(f"{masked_percent}%_masked_results.txt", "w") as f:
    f.write("Epoch\tTraining Loss\tValidation Loss\n")
    epoch_size = len(trn)
    for i in range(EPOCHS):
        start_idx = i * epoch_size
        end_idx = (i + 1) * epoch_size
        epoch_trn_loss = sum(trn_loss[start_idx:end_idx]) / epoch_size if end_idx < len(trn_loss) else sum(trn_loss[start_idx:]) / (len(trn_loss) - start_idx)
        epoch_val_loss = val_loss[i]
        f.write(f"{i+1}\t{epoch_trn_loss}\t{epoch_val_loss}\n")
    f.write(f"\nAccuracy: {acc * 100}%\n")

print(f"Результаты сохранены в файл {masked_percent}%_masked_results.txt")