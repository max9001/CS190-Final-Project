#source https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

#import pruning method
import torch.nn.utils.prune as prune
import csv
class Argument():
    def __init__(self, batch_size=64, test_batch_size=1000,epochs=3, lr=1.0,
                gamma=0.7,no_cuda=False, log_interval=100,save_model=False):
        
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.no_cuda = no_cuda
        self.log_interval = log_interval
        self.save_model = save_model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():


    args = Argument()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()
    # file = "mnist_cnn.pt"
    # model.load_state_dict(torch.load(file, map_location=device))
    # this will automatically load the file and load the parameters into the model.



    # ------- define the pruning method ------------------
    pruning_method = prune.L1Unstructured

    parameters_to_prune = [
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc1, 'weight'),
        (model.fc2, 'weight')
    ]

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method,
        amount=0.975,
    )

    # print(model.conv1.weight[:8, :4, :1,])
    # print(model.conv2.weight[:8, :4, :1,])
    # print(model.fc1.weight[:8, :8])
    # print(model.fc2.weight[:8, :8])


    sparsity = 100. * float(
        torch.sum(model.conv1.weight == 0)
        + torch.sum(model.conv2.weight == 0)
        + torch.sum(model.fc1.weight == 0)
        + torch.sum(model.fc2.weight == 0)
    ) / float(
        model.conv1.weight.nelement()
        + model.conv2.weight.nelement()
        + model.fc1.weight.nelement()
        + model.fc2.weight.nelement()
    )

    print(sparsity)

   

    test(args, model, device, test_loader)


    zero_indices = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            zero_indices[name] = torch.nonzero(module.weight == 0, as_tuple=False).cpu().numpy()

    # Save indices to a CSV file
    with open('zero_indices.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Layer', 'Index'])  # Header
        for layer_name, indices in zero_indices.items():
            for index in indices:
                writer.writerow([layer_name] + index.tolist())
    

    

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")

main()