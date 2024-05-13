import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# training set has 60,000 images and the test set has 10,000 images.
train_batch_size = 16
test_batch_size = 16

# download and load the training data
trainset = datasets.FashionMNIST('MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('MNIST_data/', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)

# use (adapted) LeNet model used earlier in this notebook
class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)  # fix input size for fc1
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), (2, 2))
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

model = LeNet().to(device)

# # define the pruning method
# pruning_percentage = 0.20
# pruning_method = prune.L1Unstructured

# parameters_to_prune = [
#     (model.conv1, 'weight'),
#     (model.conv2, 'weight'),
#     (model.fc1, 'weight'),
#     (model.fc2, 'weight'),
#     (model.fc3, 'weight')
# ]

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# 10 & 15 per instructions
# iters = 10
epochs = 10

train_losses, test_losses = [], []
accuracy_list = []
sparsity_list = []

# prune.global_unstructured(
#     parameters_to_prune,
#     pruning_method,
#     amount=0.000,
# )

# for i in range(iters):
#     if i != 0:
#         for name, param in model.named_parameters():
#             if 'weight_orig' in name:  # Only load original weights
#                 param.data = initial_state_dict[name].clone().to(device)
#         prune.global_unstructured(
#             parameters_to_prune,
#             pruning_method,
#             amount=pruning_percentage,
#         )
#     initial_state_dict = model.state_dict()
print("started training")
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Move images and labels to the appropriate device
        images, labels = images.to(device), labels.to(device)

        # Training pass
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computation
        with torch.no_grad():
            # Set the model to evaluation mode
            model.eval()

            # Validation pass
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

            model.train()

    train_losses.append(running_loss / len(trainloader))
    test_losses.append(test_loss / len(testloader))

    # sparsity = 100. * float(
    #     torch.sum(model.conv1.weight == 0)
    #     + torch.sum(model.conv2.weight == 0)
    #     + torch.sum(model.fc1.weight == 0)
    #     + torch.sum(model.fc2.weight == 0)
    #     + torch.sum(model.fc3.weight == 0)
    # ) / float(
    #     model.conv1.weight.nelement()
    #     + model.conv2.weight.nelement()
    #     + model.fc1.weight.nelement()
    #     + model.fc2.weight.nelement()
    #     + model.fc3.weight.nelement()
    # )

    # sparsity_list.append(sparsity)
    # accuracy_list.append(accuracy / len(testloader))

    print(
        "Training loss: {:.3f}..".format(running_loss / len(trainloader)),
        "Test loss: {:.3f}..".format(test_loss / len(testloader)),
        "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))
        #   "Global sparsity: {:.2f}%".format(sparsity))

    # print(model.fc3.weight[:8, :4])
    # print()

# # Plot the accuracy vs sparsity level
# plt.plot(sparsity_list, accuracy_list, marker='o')
# plt.xlabel('Global Sparsity (%)')
# plt.ylabel('Test Accuracy')
# plt.title('Accuracy vs Sparsity')
# plt.grid(True)
# plt.show()