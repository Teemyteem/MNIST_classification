# import some packages you need here
import dataset
from model import LeNet5, CustomMLP
import torch
from torchvision import transforms
import time
from torchsummary import summary
import matplotlib.pyplot as plt

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    # write your codes here
     # Set model to training mode
    model.train()

    # Initialize variables for loss and correct predictions
    tot_loss = 0.0
    pred = 0

    # Iterate over the data loader
    for inputs, labels in trn_loader:
        # Move data to the appropriate device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        tot_loss += loss.item() * inputs.size(0)

        # Compute the number of correct predictions
        _, predicted = torch.max(outputs, 1)
        pred += (predicted == labels).sum().item()


    trn_loss = tot_loss / len(trn_loader.dataset)
    acc = pred / len(trn_loader.dataset)

    return trn_loss, acc


def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    # Set model to evaluation mode
    model.eval()

    # Initialize variables for loss and correct predictions
    tot_loss = 0.0
    pred = 0

    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over the data loader
        for inputs, labels in tst_loader:
            # Move data to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Accumulate loss
            tot_loss += loss.item() * inputs.size(0)

            # Compute the number of correct predictions
            _, predicted = torch.max(outputs, 1)
            pred += (predicted == labels).sum().item()

    # Compute average loss
    tst_loss = tot_loss / len(tst_loader.dataset)

    # Compute accuracy
    acc = pred / len(tst_loader.dataset)
    
    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here
    epochs = 10
    batch_size = 64
    learning_rate = 0.01
    mommentum = 0.5

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.1307], [0.3081])])
    
    train_data = dataset.MNIST(data_dir='../data/train.tar', transform=transform, is_train=True)
    test_data = dataset.MNIST(data_dir='../data/test.tar', transform=transform, is_train=False)

    trn_loader = dataset.DataLoader(train_data, batch_size=batch_size)
    tst_loader = dataset.DataLoader(test_data, batch_size=batch_size)


#     LeNet-5 model
    lenet_model = LeNet5().to(device)
    optimizer = torch.optim.SGD(lenet_model.parameters(), lr=learning_rate, momentum=mommentum)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    trn_acc_avg, trn_loss_avg, tst_acc_avg, tst_loss_avg = [], []
    for epoch in range(epochs):
        trn_loss, trn_acc = train(model=lenet_model,
                                  trn_loader=trn_loader,
                                  device=device,
                                  criterion=criterion,
                                  optimizer=optimizer)
        tst_loss, tst_acc = test(model=lenet_model,
                                 tst_loader=tst_loader,
                                 device=device,
                                 criterion=criterion)
        
        trn_acc_avg.append(trn_acc)
        trn_loss_avg.append(trn_loss)
        tst_acc_avg.append(tst_acc)
        tst_loss_avg.append(tst_loss)

    print("LeNet-5 Model Summary")
    summary(lenet_model.to(device), (1, 28, 28))

    
#     MLP model
    mlp_model = CustomMLP().to(device)
    optimizer = torch.optim.SGD(mlp_model.parameters(), lr=learning_rate, momentum=mommentum)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    trn_acc_avg, trn_loss_avg, tst_acc_avg, tst_loss_avg = [], []
    for epoch in range(epochs):
        trn_loss, trn_acc = train(model=mlp_model,
                                  trn_loader=trn_loader,
                                  device=device,
                                  criterion=criterion,
                                  optimizer=optimizer)
        tst_loss, tst_acc = test(model=mlp_model,
                                 tst_loader=tst_loader,
                                 device=device,
                                 criterion=criterion)
        
        trn_acc_avg.append(trn_acc)
        trn_loss_avg.append(trn_loss)
        tst_acc_avg.append(tst_acc)
        tst_loss_avg.append(tst_loss)

    print("Custom Model Summary")
    summary(mlp_model.to(device), (1, 28, 28))

if __name__ == '__main__':
    main()
