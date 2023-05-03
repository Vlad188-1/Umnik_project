import torch
import pandas as pd
from datetime import datetime
from numpy import inf
from rich import print
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


def load_model(path_to_model: str):
    model = torch.load(path_to_model)
    model.eval()
    return model


def inference(path_to_config: str,
              path_to_AE_model: str,
              path_to_NN_model: str):
    x_test = pd.read_csv("../projects/for_test_models/X_valid.csv")
    # x_test = x_test.drop(["well id", "depth, ", "lith", "goal"], axis=1)

    x_test = torch.from_numpy(x_test.values).to(torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    AE_model = load_model(path_to_AE_model).to(device)
    NN_model = load_model(path_to_NN_model).to(device)

    x_test_encoder = AE_model.encoder(x_test)
    res = NN_model(x_test_encoder)
    outputs = torch.sigmoid(res).cpu()
    print(x_test_encoder)
    print(x_test_encoder.shape)
    print(res)
    print(res.shape)
    print(outputs)

    # print(NN_model)


path_to_AE_model = "../runs/base_example/best_model_AE.pt"
path_to_NN_model = "../runs/base_example/best_model_NN.pt"

inference(path_to_AE_model, path_to_NN_model)