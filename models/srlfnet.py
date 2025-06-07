import torch
import torch.nn as nn
import numpy as np

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc1_2 = nn.Linear(768, 512)
        self.fc1_3 = nn.Linear(768, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn1_2 = nn.BatchNorm1d(512)
        self.bn1_3 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.relu1_2 = nn.ReLU()
        self.relu1_3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_2 = nn.Linear(512, 256)
        self.fc2_3 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn2_2 = nn.BatchNorm1d(256)
        self.bn2_3 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.relu2_2 = nn.ReLU()
        self.relu2_3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.6)
        self.fcRes = nn.Linear(256 * 3, 256 * 3)
        self.bnRes = nn.BatchNorm1d(256 * 3)
        self.reluRes = nn.ReLU()
        self.fc3 = nn.Linear(256 * 3, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(128, 16)

    def forward(self, x, x_2, x_3):
        x = x.view(-1, 768)
        x_2 = x_2.view(-1, 768)
        x_3 = x_3.view(-1, 768)
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x_2 = self.dropout1(self.relu1_2(self.bn1_2(self.fc1_2(x_2))))
        x_3 = self.dropout1(self.relu1_3(self.bn1_3(self.fc1_3(x_3))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x_2 = self.dropout2(self.relu2_2(self.bn2_2(self.fc2_2(x_2))))
        x_3 = self.dropout2(self.relu2_3(self.bn2_3(self.fc2_3(x_3))))
        x = torch.cat((x, x_2, x_3), dim=1)
        identity = x
        x = self.fcRes(x)
        x = self.bnRes(x)
        x = self.reluRes(x) + identity
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.relu4(self.bn4(self.fc4(x)))
        x = self.relu5(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        return torch.softmax(x, dim=1)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.dash = np.load(data_path + '/total_dashboard_data.npz')['embeddings']
        self.rear = np.load(data_path + '/total_rearview_data.npz')['embeddings']
        self.side = np.load(data_path + '/total_side_data.npz')['embeddings']
        self.labels = np.load(data_path + "/total_labels.npz")['labels']

        min_length = min(len(self.dash), len(self.rear), len(self.side), len(self.labels))
        self.dash = self.dash[:min_length]
        self.rear = self.rear[:min_length]
        self.side = self.side[:min_length]
        self.labels = self.labels[:min_length]

        class_0_and_1_indices = np.where((self.labels == 0) | (self.labels == 1))[0]
        thinned_indices = class_0_and_1_indices[::1]
        other_class_indices = np.where((self.labels != 0) & (self.labels != 1))[0]
        keep_indices = np.sort(np.concatenate((thinned_indices, other_class_indices)))

        self.dash = self.dash[keep_indices]
        self.rear = self.rear[keep_indices]
        self.side = self.side[keep_indices]
        self.labels = self.labels[keep_indices]
        self.labels = np.where(self.labels == 1, 0, self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = (torch.Tensor(self.dash[idx]), torch.Tensor(self.rear[idx]), torch.Tensor(self.side[idx]))
        label = self.labels[idx] - 1 if self.labels[idx] > 0 else self.labels[idx]
        return (*embedding, torch.Tensor([label]).long())

    def get_class_counts(self):
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        return unique_labels, counts

class CustomDatasetTest(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.dash = np.load(data_path + '/total_dashboard_data.npz')['embeddings']
        self.rear = np.load(data_path + '/total_rearview_data.npz')['embeddings']
        self.side = np.load(data_path + '/total_side_data.npz')['embeddings']
        self.labels = np.load(data_path + "/total_labels.npz")['labels']

        class_0_and_1_indices = np.where((self.labels == 0) | (self.labels == 1))[0]
        thinned_indices = class_0_and_1_indices[::1]
        other_class_indices = np.where((self.labels != 0) & (self.labels != 1))[0]
        keep_indices = np.sort(np.concatenate((thinned_indices, other_class_indices)))

        self.dash = self.dash[keep_indices]
        self.rear = self.rear[keep_indices]
        self.side = self.side[keep_indices]
        self.labels = self.labels[keep_indices]
        self.labels = np.where(self.labels == 1, 0, self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = (torch.Tensor(self.dash[idx]), torch.Tensor(self.rear[idx]), torch.Tensor(self.side[idx]))
        label = self.labels[idx] - 1 if self.labels[idx] > 0 else self.labels[idx]
        return (*embedding, torch.Tensor([label]).long())

    def get_class_counts(self):
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        return unique_labels, counts
