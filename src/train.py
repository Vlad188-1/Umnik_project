import torch
from datetime import datetime
from numpy import inf
from rich import print
from pathlib import Path

# My imports
from src.NN import CustomDataset, DataLoader
from utils.PlotGraphics import plotTrainValidCurve, plotTrainValidCurveAE
from src.NN import NN, AutoEncoder


def train_and_validation(model,
                         train_dataset: CustomDataset,
                         val_dataset: CustomDataset,
                         batch_size: int,
                         epochs: int,
                         device: str,
                         criterion: torch.nn.modules.loss,
                         optimizer: torch.optim.Adam,
                         out_dir: str,
                         scheduler=None):
    all_train_loss = []
    all_val_loss = []
    all_train_accuracy = []
    all_val_accuracy = []
    best_loss = inf

    # criterion = nn.CrossEntropyLoss(reduction="sum")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)

    start = datetime.now()
    # Train
    for epoch in range(1, epochs + 1):

        train_loss = 0
        train_acc = 0

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            labels = labels.reshape(labels.shape[0])

            optimizer.zero_grad()
            output = model(inputs)

            _, preds = torch.max(output, 1)

            loss = criterion(output, labels) #.reshape(labels.shape[0]))
            #loss = criterion(torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)[0].unsqueeze(1), labels)
            # acc = binary_acc(output, labels)
            # acc = multi_acc(output, labels)
            acc = torch.sum(preds == labels.data) / inputs.size(0)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()

        train_loss_per_batch = train_loss / len(train_dataloader)
        train_accuracy_per_batch = train_acc / len(train_dataloader)

        all_train_loss.append(train_loss_per_batch)
        all_train_accuracy.append(train_accuracy_per_batch)

        # Validation
        val_loss = 0
        val_acc = 0

        model.eval()
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            labels = labels.reshape(labels.shape[0])

            outputs = model(inputs)

            # loss = criterion(torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)[0].unsqueeze(1), labels)
            # loss = criterion(output, labels.reshape(labels.shape[0]))
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            acc = torch.sum(preds == labels.data) / inputs.size(0)
            # acc = binary_acc(output, labels)
            # acc = multi_acc(output, labels)

            val_loss += loss.item()
            val_acc += acc.item()

        val_loss_per_batch = val_loss / len(val_dataloader)
        val_accuracy_per_batch = val_acc / len(val_dataloader)

        all_val_loss.append(val_loss_per_batch)
        all_val_accuracy.append(val_accuracy_per_batch)
        # scheduler.step(val_loss)

        print(f"Epoch = [bold red]{epoch}/{epochs}[/bold red] | Train_loss = [bold yellow]{train_loss_per_batch:.5f} [/bold yellow] \
| Accuracy_train = [bold yellow]{train_accuracy_per_batch:.5f}[/bold yellow] \
| Val_loss = [bold green]{val_loss_per_batch:.5f}[/bold green] | Accuracy_val = [bold green]{val_accuracy_per_batch:.5f}[/bold green]",
              end="\r")

        if val_loss_per_batch < best_loss:
            best_loss = val_loss_per_batch
            model_path = Path(out_dir, f'best_model_NN.pt')
            torch.save(model, model_path)

        history = {"train_loss": all_train_loss,
                   "val_loss": all_val_loss,
                   "train_accuracy": all_train_accuracy,
                   "val_accuracy": all_val_accuracy
                   }
        if epoch % 5 == 0:
            plotTrainValidCurve(history, out_dir)
    end = datetime.now()
    print("Время обучения полносвязной сети: ", end - start)
    #
    # return history


def train_and_validation_autoencoder(model_AE: AutoEncoder,
                                     model_NN: NN,
                                     train_dataset: CustomDataset,
                                     val_dataset: CustomDataset,
                                     batch_size: int,
                                     epochs_AE: int,
                                     epochs_NN: int,
                                     device: str,
                                     criterion_AE: torch.nn.modules.loss,
                                     criterion_NN: torch.nn.modules.loss,
                                     optimizer_AE: torch.optim.Adam,
                                     optimizer_NN: torch.optim.Adam,
                                     out_dir: str,
                                     scheduler=None):
    all_train_AE_loss = []
    all_val_AE_loss = []
    best_loss = inf

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model_AE = model_AE.to(device)
    print("Начало обучения автоэкнодера...")

    # Train
    start = datetime.now()
    for epoch in range(1, epochs_AE + 1):

        train_loss = 0

        for inputs, _ in train_dataloader:
            inputs = inputs.to(device)

            optimizer_AE.zero_grad()
            output = model_AE(inputs)

            loss = criterion_AE(output, inputs)
            loss.backward()
            optimizer_AE.step()
            train_loss += loss.item()
        train_loss_per_batch = train_loss / len(train_dataloader)
        all_train_AE_loss.append(train_loss_per_batch)

        # Validation
        val_loss = 0

        model_AE.eval()
        for inputs, _ in val_dataloader:
            inputs = inputs.to(device)

            output = model_AE(inputs)
            loss = criterion_AE(output, inputs)
            val_loss += loss.item()
        val_loss_per_batch = val_loss / len(val_dataloader)
        all_val_AE_loss.append(val_loss_per_batch)

        print(f"Epoch = [bold red]{epoch}/{epochs_AE}[/bold red] | Train_loss = [bold yellow]{train_loss_per_batch:.5f} [/bold yellow] \
    | Val_loss = [bold green]{val_loss_per_batch:.5f}[/bold green]", end="\r")

        if val_loss_per_batch < best_loss:
            best_loss = val_loss_per_batch
            model_path = Path(out_dir, f'best_model_AE.pt')
            torch.save(model_AE, model_path)

        history = {"train_loss": all_train_AE_loss,
                   "val_loss": all_val_AE_loss,
                   }
        if epoch % 5 == 0:
            plotTrainValidCurveAE(history, out_dir)
    print("")
    end = datetime.now()
    print("Время обучения автоэнкодера: ", end - start)

    # Train simple NN
    model_encoder = model_AE.encoder.to(device)

    with torch.no_grad():
        x_train_encoder = model_encoder(train_dataset.x_data)
        x_val_encoder = model_encoder(val_dataset.x_data)

    train_dataset_encoder = CustomDataset(x_train_encoder, train_dataset.y_data)
    val_dataset_encoder = CustomDataset(x_val_encoder, val_dataset.y_data)

    # model_NN = NN(in_features=x_train_encoder.shape[1], num_classes=1).to(device)

    # optimizer = Adam(model_NN.parameters(), lr=0.0001)

    print("[bold white]Начало обучения полносвязной сети...")
    train_and_validation(model=model_NN,
                         train_dataset=train_dataset_encoder,
                         val_dataset=val_dataset_encoder,
                         batch_size=batch_size,
                         epochs=epochs_NN,
                         device=device,
                         criterion=criterion_NN,
                         optimizer=optimizer_NN,
                         out_dir=out_dir)
