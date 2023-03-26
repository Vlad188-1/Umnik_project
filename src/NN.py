import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from collections import OrderedDict


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


class AutoEncoder(nn.Module):

    def __init__(self, in_features):
        super(AutoEncoder, self).__init__()
        self.in_features = in_features

        out_layer_1_features = self.in_features - 2
        out_layer_2_features = out_layer_1_features - 2
        out_layer_3_features = out_layer_2_features - 2

        self.encoder = nn.Sequential(
            OrderedDict([
                ("linear1", nn.Linear(in_features=self.in_features, out_features=out_layer_1_features)),
                ("relu1", nn.ReLU()),
                ("batchnorm1", nn.BatchNorm1d(out_layer_1_features)),
                ("linear2", nn.Linear(in_features=out_layer_1_features, out_features=out_layer_2_features)),
                ("relu2", nn.ReLU()),
                ("batchnorm2", nn.BatchNorm1d(out_layer_2_features)),
                ("bottleneck_linear3", nn.Linear(in_features=out_layer_2_features, out_features=out_layer_3_features)),
                ("batchnorm3", nn.BatchNorm1d(out_layer_3_features)),
                ])
        )
        self.decoder = nn.Sequential(
            OrderedDict([
                ("linear1", nn.Linear(in_features=out_layer_3_features, out_features=out_layer_2_features)),
                ("relu1", nn.ReLU()),
                ("batchnorm1", nn.BatchNorm1d(out_layer_2_features)),
                ("linear2", nn.Linear(in_features=out_layer_2_features, out_features=out_layer_1_features)),
                ("relu2", nn.ReLU()),
                ("batchnorm2", nn.BatchNorm1d(out_layer_1_features)),
                ("linear3", nn.Linear(in_features=out_layer_1_features, out_features=self.in_features))
            ])
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
