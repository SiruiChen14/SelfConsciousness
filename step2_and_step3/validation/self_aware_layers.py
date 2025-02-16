import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import argparse

base_path = os.path.dirname(os.path.dirname(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='llama3.1_8B_instruct')
parser.add_argument('--dataset_name', type=str, default='situational_awareness')
args = parser.parse_args()

exp_name = args.model_name + '_' + args.dataset_name
save_path = os.path.join(base_path, 'exp', exp_name + '_layer')
models_save_dir = os.path.join(save_path, "models")

os.makedirs(save_path, exist_ok=True)
os.makedirs(models_save_dir, exist_ok=True)

features = np.load(os.path.join(base_path, 'features', exp_name + '_layer_wise.npy'))
labels = np.load(os.path.join(base_path, 'features', exp_name + '_labels.npy'))

features = torch.tensor(features, dtype=torch.float32).to(device)
labels = torch.tensor(labels, dtype=torch.long).to(device)

num_layers = features.shape[1]

dataset = TensorDataset(features, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=2000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2000, shuffle=False)


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


def train_and_evaluate(train_loader, test_loader, layer_index, model_save_path):
    model = SimpleClassifier(features.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0.0

    for epoch in range(20):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs = inputs[:, layer_index, :]
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs[:, layer_index, :]
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{20}, layer {layer_index} accuracy: {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_save_path)

    return best_accuracy


accuracies = []
for i in range(num_layers):
    model_save_path = os.path.join(models_save_dir, f'model_dim_{i}.pth')
    accuracy = train_and_evaluate(train_loader, test_loader, i, model_save_path)
    print(f"Dimension {i} accuracy: {accuracy}")
    accuracies.append(accuracy)

with open(os.path.join(save_path, exp_name + '_layer_wise_accuracies.json'), 'w') as json_file:
    json.dump(accuracies, json_file)
