import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def prune_weights(model, threshold):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                print(f'Param max: {torch.max(param)}')
                print(f'Param min: {torch.min(param)}')
                mask = torch.abs(param) > threshold
                # mask = torch.zeros_like(param)
                param.mul_(mask)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Model, Loss, and Optimizer
model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# Training the model
epochs = 5
for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} - Training loss: {running_loss/len(trainloader)}")


def test_model_acc(post_prune=False):
    # Testing the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(100 * correct / total)
    # print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')
    print(f'Accuracy of the model on the 10000 test images ' + ('post' if post_prune else 'pre') + f'-pruning: {100 * correct / total}%')


# test_model_acc()
# # Apply pruning
# prune_weights(model, threshold=0.8)
# test_model_acc(post_prune=True)




import torch

# Specify the size of the tensor, e.g., a 3x3 tensor
size = (3, 3)

# Create a tensor where each element is 1 with probability 0.5 and 0 otherwise
tensor = torch.bernoulli(torch.full(size, 0.5))

print(tensor)