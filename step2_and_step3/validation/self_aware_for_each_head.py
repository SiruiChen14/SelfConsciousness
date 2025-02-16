import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import argparse
from einops import rearrange

base_path = os.path.dirname(os.path.dirname(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='llama3.1_8B_instruct')
parser.add_argument('--dataset_path', type=str, default=os.path.join(base_path, 'data'))
args = parser.parse_args()

num_heads_dict = {
    'llama3.1_8B_instruct': 32,
    'llama3.1_70B_instruct': 64,
    'internlm2_5-20b-chat': 48,
    'Mistral-Nemo-Instruct-2407': 32
}

for data_set_name in os.listdir(args.dataset_path):
    data_set_name = data_set_name[:-5]
    model_name = args.model_name + '_' + data_set_name
    num_heads = num_heads_dict[args.model_name]
    data_path = os.path.join(base_path, 'features')
    save_path = os.path.join(base_path, 'exp_each_head', model_name)
    save_dir = os.path.join(save_path, model_name + '_heads' + "_models")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    features = np.load(os.path.join(data_path, model_name + '_head_wise.npy'))
    labels = np.load(os.path.join(data_path, model_name + '_labels.npy'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = torch.tensor(features, dtype=torch.float32).to(device)
    features = rearrange(features, 'b l (h d) -> b l h d', h=num_heads)
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


    def train_and_evaluate(train_loader, test_loader, layer_index, head_index, model_save_path):
        model = SimpleClassifier(features.shape[3]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_accuracy = 0.0

        for epoch in range(20):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                inputs = inputs[:, layer_index, head_index, :]
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs[:, layer_index, head_index, :]
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            accuracy = correct / total
            print(f"Epoch {epoch + 1}/{20}, layer {layer_index}, head {head_index} accuracy: {accuracy}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), model_save_path)

        return best_accuracy


    accuracies = []
    for i in range(num_layers):
        for j in range(num_heads):
            model_save_path = os.path.join(save_dir, f'model_layer_{i}_head_{j}.pth')
            accuracy = train_and_evaluate(train_loader, test_loader, i, j, model_save_path)
            print(f"Best accuracy for layer {i} head {j}: {accuracy}")
            accuracies.append(accuracy)

    with open(os.path.join(save_path, model_name + '_head_wise_accuracies.json'), 'w') as json_file:
        json.dump(accuracies, json_file)

    num_layers = int(np.ceil(len(accuracies) / num_heads))
    accuracies = np.array(accuracies).reshape(num_layers, num_heads)
