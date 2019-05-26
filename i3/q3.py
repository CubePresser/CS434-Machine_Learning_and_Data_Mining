
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


class Net(nn.Module):
    def __init__(self, dropout):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32, 100)
        self.fc1_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(100, 100)
        self.fc2_drop = nn.Dropout(dropout)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32)
        x = torch.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = torch.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)

print('Using PyTorch version:', torch.__version__, ' Device:', device)

# A class that we can use to introduce new values of momentum, dropout and weight decay
# through iterative loops later in the code to generate multiple plots per single run
class Q3:
    # momentum = None
    # dropout = None
    # weightdecay = None
    def __init__(self, mom, do, wd, name, index):
        momentum = mom
        dropout = do
        weightdecay = wd
        name = name
        index = index
        learningRate = 0.1
        #momentum = float(sys.argv[1]) #0.5
        #dropout = float(sys.argv[2]) #0.2
        #weightdecay = float(sys.argv[3]) #0

        model = Net(dropout).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentum, weight_decay=weightdecay)
        criterion = nn.CrossEntropyLoss()

        print(model)

        batch_size = 32
        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

        train_dataset = datasets.CIFAR10('./data', 
                                    train=True, 
                                    download=True, 
                                    transform=transform)

        # train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])
        # validation_dataset = torch.utils.data.Subset(train_dataset, [])
        validation_dataset = datasets.CIFAR10('./data', 
                                            train=True, 
                                            transform=transform)

        testing_dataset = datasets.CIFAR10('./data',
                                            train=False,
                                            transform=transform,
                                            download=True)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = 40000

        train_idx, valid_idx = indices[:split], indices[split:]
        train_dataset = Subset(train_dataset, train_idx)
        validation_dataset = Subset(validation_dataset, valid_idx)


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
            
            return accuracy



        #%%time
        epochs = 5

        lossv, accv = [], []
        for epoch in range(1, epochs + 1):
            train(epoch)
            validate(lossv, accv, validation_loader)

        testv, taccv = [], []
        print("THE BELOW IS TEST ACCURACY")
        self.accr = validate(testv, taccv, testing_loader)
        
        # Generate plot of loss as a function of training epochs
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(np.arange(1,epochs+1), lossv, label="Loss")
        plt.title('Validation Loss')
        plt.legend(loc='best')
        plt.savefig('q3plot1' + name + str(index))

        # Generate plot of accuracy as a function of training epochs
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Percentage')
        plt.plot(np.arange(1,epochs+1), accv, label="Accuracy")
        plt.title('Validation Accuracy')
        plt.legend(loc='best')
        plt.savefig('q3plot2' + name + str(index))

Q3(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5]))

###
### ORIGINAL CODE FOR RUNNING ALL M, D and WD values into plots
###

# m  = [0.25, 0.5, 0.75]
# d  = [0.2, 0.4, 0.6]
# wd = [0, 0.1, 0.01]

# ms = []
# for i in range(3):
#     q = Q3(m[i], d[0], wd[0], "m", i)
#     ms.append(q.accr)
# #Generate plot with ms
# plt.figure()
# plt.xlabel('m')
# plt.ylabel('Accuracy')
# plt.plot(np.arange(1,3), ms, label="m-accuracy")
# plt.title('Accuracy with m')
# plt.legend(loc='best')
# plt.savefig('q3Mplot')

# ds = []
# for k in range(3):
#     q = Q3(m[1], d[k], wd[0], "d", k)
#     ds.append(q.accr)
# #Generate plot with ds
# plt.figure()
# plt.xlabel('d')
# plt.ylabel('Accuracy')
# plt.plot(np.arange(1,3), ds, label="d-accuracy")
# plt.title('Accuracy with d')
# plt.legend(loc='best')
# plt.savefig('q3Dplot')

# wds = []
# for j in range(3):
#     q = Q3(m[1], d[0], wd[j], "wd", j)
#     wds.append(q.accr)
# #Generate plot with wds
# plt.figure()
# plt.xlabel('wd')
# plt.ylabel('Accuracy')
# plt.plot(np.arange(1,3), wds, label="wd-accuracy")
# plt.title('Accuracy with wd')
# plt.legend(loc='best')
# plt.savefig('q3WDplot')