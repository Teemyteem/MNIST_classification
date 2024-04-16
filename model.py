import torch.nn as nn
import torch

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):

        # write your codes here
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()
       

    def forward(self, img):

        # write your codes here
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool2(output)
        output = output.view(output.shape[0], -1)
        
        output = self.fc1(output)
        output = self.relu3(output)
        output = self.fc2(output)
        output = self.relu4(output)
        output = self.fc3(output)
        output = self.relu5(output)

        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):

        # write your codes here
        super().__init__()
        
        self.fc1 = nn.Linear(784, 784)
        self.dropout1 = nn.Dropout(p=0.5)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(784, 64)
        self.dropout2 = nn.Dropout(p=0.5)
        self.relu2 = nn.ReLU()
        
        self.fc4 = nn.Linear(64, 32)
        self.dropout4 = nn.Dropout(p=0.5)
        self.relu4 = nn.ReLU()
        
        self.fc5 = nn.Linear(32, 16)
        self.dropout5 = nn.Dropout(p=0.5)
        self.relu5 = nn.ReLU()
        
        self.fc6 = nn.Linear(16, 10)
        self.dropout6 = nn.Dropout(p=0.5)
        self.relu6 = nn.ReLU()   
        
        self.fc8 = nn.Linear(10, 10)
        self.softmax1 = nn.Softmax(1)

    def forward(self, img):

        # write your codes here  
        output = img.view(-1, 28 * 28)
        output = self.fc1(output)
        output = self.dropout1(output)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.dropout2(output)
        output = self.relu2(output)
        output = self.fc4(output)
        output = self.dropout4(output)
        output = self.relu4(output)
        output = self.fc5(output)
        output = self.dropout5(output)
        output = self.relu5(output)
        output = self.fc6(output)
        output = self.dropout6(output)
        output = self.relu6(output)
        output = self.fc8(output)
        output = self.softmax1(output)
   
        return output
