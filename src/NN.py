import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class CustomDataset(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


class NN(nn.Module):

    def __init__(self, in_features, num_classes):
        super(NN, self).__init__()
        self.in_features = in_features

        self.layer_1 = nn.Linear(in_features=self.in_features, out_features=128)
        self.dropout_1 = nn.Dropout(p=0.25)
        self.layer_2 = nn.Linear(in_features=128, out_features=256)
        self.dropout_2 = nn.Dropout(p=0.35)
        self.layer_3 = nn.Linear(in_features=256, out_features=128)
        self.dropout_3 = nn.Dropout(p=0.4)
        self.layer_4 = nn.Linear(in_features=128, out_features=64)
        self.dropout_4 = nn.Dropout(p=0.5)
        self.layer_5 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.dropout_1(x)

        x = self.layer_2(x)
        x = F.relu(x)
        x = self.dropout_2(x)

        x = self.layer_3(x)
        x = F.relu(x)
        x = self.dropout_3(x)

        x = self.layer_4(x)
        x = F.relu(x)
        x = self.dropout_4(x)

        x = self.layer_5(x)
        return x
