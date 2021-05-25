import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
import tqdm
from torch.nn import ModuleList
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat

batch_size = 64
learning_rate = 0.001
    
class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def swap_and_crop(x):
    data = np.transpose(x, (3, 2, 0, 1))
    data = torch.Tensor(data)
    data = nn.functional.interpolate(data, size = (28, 28))
    return data
    
def label_to_tensor(y):
    data = torch.Tensor(y)
    data = torch.squeeze(data)
    data[data == 10] = 0
    return data.type(torch.long)

train_file = loadmat('train_32x32.mat')
test_file = loadmat('test_32x32.mat')

svhn_train_x = swap_and_crop(train_file['X'])
svhn_train_y = label_to_tensor(train_file['y'])

svhn_test_x = swap_and_crop(test_file['X'])
svhn_test_y = label_to_tensor(test_file['y'])

train_data_SVHN = CustomDataset(svhn_train_x, svhn_train_y)
test_data_SVHN = CustomDataset(svhn_test_x, svhn_test_y)

svhn_train_set, svhn_val_set = torch.utils.data.random_split(train_data_SVHN, [63257, 10000])

svhn_train_loader = torch.utils.data.DataLoader(svhn_train_set, batch_size = batch_size, shuffle = True)
svhn_dev_loader = torch.utils.data.DataLoader(svhn_val_set, batch_size = batch_size)
svhn_test_loader = torch.utils.data.DataLoader(test_data_SVHN, batch_size = batch_size)


class ModelClass(nn.Module):
    """
    TODO: Write down your model
    """
    def __init__(self):
        super(ModelClass, self).__init__()
        self.keep_prob = 0.5
        self.conv1 = nn.Conv2d(3, 32, 3, stride = 1, padding = 0, bias = False)
        self.conv2 = nn.Conv2d(32, 64, 5, stride = 1, padding = 0, bias = False)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024, 120, bias = True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(120, 84, bias = True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(84, 10, bias = True)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        
        x = torch.nn.functional.relu(self.fc1(x))
        torch.nn.Dropout(p = 1 - self.keep_prob)
        x = torch.nn.functional.relu(self.fc2(x))
        torch.nn.Dropout(p = 1 - self.keep_prob)
        
        x = self.fc3(x)
        return x


model = ModelClass()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
def train(model, criterion, optimizer, training_epochs = 3):
    model.train()
    for epoch in range(training_epochs):
        cost = 0
        n_batches = 0
        for X, Y in tqdm.tqdm(svhn_train_loader):
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, Y)
            loss.backward()
            optimizer.step()
            cost += loss.item()
            n_batches += 1
            
        cost /= n_batches
    print('[Epoch : {:>4}] cost = {:>.9}'.format(epoch + 1, cost))
    return cost
    
train(model, criterion, optimizer, training_epochs = 3)


def test(data_loader, model):
    model.eval()
    n_predict = 0
    n_correct = 0
    with torch.no_grad():
        for X, Y in tqdm.tqdm(data_loader):
            y_hat = model(X)
            y_hat.argmax()
            
            _, predicted = torch.max(y_hat, 1)
        
            n_predict += len(predicted)
            n_correct += (Y == predicted).sum()
            
    accuracy = n_correct / n_predict
    print(f"Accuracy : {accuracy} ()")

test(svhn_dev_loader, model)
