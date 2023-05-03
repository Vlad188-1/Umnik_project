import torch
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import numpy as np
from utils.plots import plot_curve_testing, plot_test_markup
from rich import print
import yaml


def load_model(path_to_model: str):
    model = torch.load(path_to_model)
    model.eval()
    return model


def test(path_to_AE_model: str,
         path_to_NN_model: str,
         path_to_config: str,
         data: pd.DataFrame):

    for i in tqdm(data.columns, desc='Read files...'):
        if i == 'lith':
            continue
        else:
            data[i] = data[i].replace(',', '.', regex=True).astype(np.float32)

    # Preprocess using config
    with open(path_to_config, "r") as yml:
        config = yaml.safe_load(yml)

    for k in sorted(config.keys()):
        if k == "^2":
            for feature in config[k]:
                data[feature + "^2"] = data[feature]**2
        elif k == "^3":
            for feature in config[k]:
                data[feature + "^3"] = data[feature]**3
        elif k == "*":
            for feature in config[k]:
                if len(feature) == 2:
                    data["*".join(feature)] = data[feature[0]] * data[feature[1]]
                elif len(feature) == 3:
                    data["*".join(feature)] = data[feature[0]] * data[feature[1]] * data[feature[2]]
                elif len(feature) == 4:
                    data["*".join(feature)] = data[feature[0]] * data[feature[1]] * data[feature[2]] \
                                                   * data[feature[3]]
    data.dropna(inplace=True)
    x_test = data.drop(["well id", "depth, m", "lith"], axis=1)
    x_test.dropna(inplace=True)
    x_test.sort_index(axis=1, inplace=True)

    y_test = x_test["goal"].values
    x_test = x_test.drop(["goal"], axis=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load models
    AE_model = load_model(path_to_AE_model).to(device)
    NN_model = load_model(path_to_NN_model).to(device)

    x_test = torch.from_numpy(x_test.values).to(torch.float32)
    x_test_encoder = AE_model.encoder(x_test)
    outputs = NN_model(x_test_encoder)
    y_pred = torch.sigmoid(outputs).cpu().detach().numpy()

    print(f1_score(y_test, y_pred.round()))
    print(classification_report(y_test, y_pred.round(), target_names=["no_oil", "oil"]))

    # Plot precision-reall-curve
    print("Построение Precision-Recall кривой")
    # assert y_test.shape == y_pred
    plot_curve_testing(y_test, y_pred)
    plot_test_markup(data, y_test, y_pred)


path_to_AE_model = "../projects/proba_sorted_features_2/best_model_AE.pt"
path_to_NN_model = "../projects/proba_sorted_features_2/best_model_NN.pt"
path_to_config = "../projects/proba_sorted_features_2/config_processing_data.yml"

data = pd.read_csv("../test_wells/well_23.csv")

test(path_to_AE_model, path_to_NN_model, path_to_config, data)