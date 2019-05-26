
#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32, 100)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 100)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32)
        x = torch.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = torch.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)

# Get learning rate from command line
learningRate = float(sys.argv[1])
momentum = 0.5

model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

print(model)



batch_size = 32
transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

train_dataset = datasets.CIFAR10('./data', 
                               train=True, 
                               download=True, 
                               transform=transform)

validation_dataset = datasets.CIFAR10('./data', 
                                    train=True, 
                                    transform=transform)

testing_dataset = datasets.CIFAR10('./data',
                                    train=False,
                                    transform=transform,
                                    download=True)

# Get all training set indices
num_train = len(train_dataset)
indices = list(range(num_train))
split = 40000

# Generate subsets of the training set into 4 batches of training and 1 batch for validation
train_idx, valid_idx = indices[:split], indices[split:]
train_dataset = Subset(train_dataset, train_idx)
validation_dataset = Subset(validation_dataset, valid_idx)

#Load data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                                batch_size=batch_size)

testing_loader = torch.utils.data.DataLoader(dataset=testing_dataset,
                                                batch_size=batch_size)

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break
print('validationsize: ' + str(len(validation_loader)))
print('trainsize: ' + str(len(train_loader)))

def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def validate(loss_vector, accuracy_vector, loader):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(loader.dataset), accuracy))



#%%time
epochs = 5

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv, validation_loader)

testv, taccv = [], []
print("THE BELOW IS TEST ACCURACY")
validate(testv, taccv, testing_loader)

# Generate plot of loss as a function of training epochs
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(np.arange(1,epochs+1), lossv, label="Loss")
plt.title('Validation Loss')
plt.legend(loc='best')
plt.savefig('3q2plot1')

# Generate plot of accuracy as a function of training epochs
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Accuracy Percentage')
plt.plot(np.arange(1,epochs+1), accv, label="Accuracy")
plt.title('Validation Accuracy')
plt.legend(loc='best')
plt.savefig('3q2Plot2')
