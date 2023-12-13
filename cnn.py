import pandas as pd
import numpy as np
import torch
import cv2
import torchvision
import torchvision.transforms as transforms
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

X_train = pd.read_csv(sys.argv[1] + "/train_x.csv")
Y_train = pd.read_csv(sys.argv[1] + "/train_y.csv")

m = X_train.shape[0]

X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()

Y_train = Y_train[ :, 1]

X_test = pd.read_csv(sys.argv[1] + "/non_comp_test_x.csv")
Y_test = pd.read_csv(sys.argv[1] + "/non_comp_test_y.csv")

m_test = X_test.shape[0]

X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()

Y_test = Y_test[:, 1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def dataloader(X, Y, batch_num, batch_size, checkacc = False):
    for i in range(batch_size):
        image_name = X[batch_num * batch_size + i][1]
        path = sys.argv[1] + '/' + image_name
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = transform(image)
        tensor = tensor.unsqueeze(0)
        if i == 0:
            tensors = tensor
        else:
            tensors = torch.cat((tensors, tensor), dim = 0)
        #print(tensors.size(), tensor.size())
    labels = torch.tensor(Y[batch_num * batch_size : (batch_num + 1) * batch_size])
    if checkacc:
        outputs = net(tensors.to(device))
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted.size(), labels.size(), sep = " ")
        return (predicted.to(device) == labels.to(device)).sum().item()
    else:
        return tensors, labels

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(73728, 128)
        self.fc2 = nn.Linear(128, 30)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 5e-5)

batch_size = 8

for epoch in range(20):
    for i in range(m // batch_size):
        X,Y = dataloader(X_train, Y_train, i, batch_size, False)
        X = X.to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        output = net(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()

predictions = {'Id' : [], 'Genre' : []}
with torch.no_grad():
    for i in range(m_test):
        pred = dataloader(X_test, Y_test, i, 1, True)
        predictions['Id'].append(i)
        predictions['Genre'].append(pred)
df = pd.DataFrame(predictions)
df.to_csv(sys.argv[1] + "/non_comp_test_pred_y.csv", index = False)
