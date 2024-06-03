# import pandas as pd
# import matplotlib.pyplot as plt
# import csv
# import numpy as np

# csvfile = 'zero_indices.csv'

# zero_indices = {}

# with open('zero_indices.csv', 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         next(reader)  # Skip header
#         for row in reader:
#             layer_name = row[0]
#             index = tuple(map(int, row[1:]))
#             if layer_name not in zero_indices:
#                 zero_indices[layer_name] = []
#             zero_indices[layer_name].append(index)

# layer_names = ['conv1', 'conv2', 'fc1', 'fc2']
# counts = [len(zero_indices.get(layer, [])) for layer in layer_names]

# plt.figure(figsize=(10, 6))
# bars = plt.bar(layer_names, counts, color='blue')

# for bar, count in zip(bars, counts):
#     yvalprint(f"The percentage difference between the two CSV files is {percentage_diff:.2f}%") = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, count, ha='center', va='bottom')

# plt.xlabel('Layer')
# plt.ylabel('Number of Zero Entries')
# plt.title('Number of Zero Entries in Each Layer')
# plt.ylim(0, max(counts) + 10)  
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import csv
import torch
import torch.nn as nn

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

# Instantiate the model
model = Net()

# Read the CSV file and create the zero_indices dictionary
csvfile = 'zero_indices.csv'
zero_indices = {}

with open(csvfile, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        layer_name = row[0]
        index = tuple(map(int, row[1:]))
        if layer_name not in zero_indices:
            zero_indices[layer_name] = []
        zero_indices[layer_name].append(index)

# Get total number of weights in each layer
total_weights = {
    'conv1': model.conv1.weight.nelement(),
    'conv2': model.conv2.weight.nelement(),
    'fc1': model.fc1.weight.nelement(),
    'fc2': model.fc2.weight.nelement()
}

# Calculate the ratio of zero entries to total weights for each layer
layer_names = ['conv1', 'conv2', 'fc1', 'fc2']
ratios = [len(zero_indices.get(layer, [])) / total_weights[layer] for layer in layer_names]

# Plotting the bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(layer_names, ratios, color='blue')

# Add text labels above each bar to show the ratio
for bar, ratio in zip(bars, ratios):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{ratio:.2f}', ha='center', va='bottom')

plt.xlabel('Layer')
plt.ylabel('Ratio of Zero Entries')
plt.title('Ratio of Zero Entries to Total Weights in Each Layer')
plt.ylim(0, max(ratios) + 0.05)  # Add some space above the tallest bar for the text
plt.show()
