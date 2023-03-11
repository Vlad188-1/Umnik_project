import torch
from datetime import datetime
from numpy import inf
from rich import print
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
# My imports
from src.NN import CustomDataset, DataLoader
from src.Metrics import binary_acc
from utils.PlotGraphics import plotTrainValidCurve


def train_and_validation(model,
                         train_dataset: CustomDataset,
                         val_dataset: CustomDataset,
                         batch_size: int,
                         epochs: int,
                         device: str,
                         criterion: torch.nn.modules.loss,
                         optimizer: torch.optim.Adam,
                         scheduler=None):

    all_train_loss = []
    all_val_loss = []
    all_train_accuracy = []
    all_val_accuracy = []
    best_loss = inf

    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    writer = SummaryWriter()

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    # Train
    for epoch in range(1, epochs + 1):

        train_loss = 0
        train_acc = 0

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(inputs)

            loss = criterion(output, labels)
            acc = binary_acc(output, labels)

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

            output = model(inputs)

            loss = criterion(output, labels)
            acc = binary_acc(output, labels)

            val_loss += loss.item()
            val_acc += acc.item()

        val_loss_per_batch = val_loss / len(val_dataloader)
        val_accuracy_per_batch = val_acc / len(val_dataloader)

        all_val_loss.append(val_loss_per_batch)
        all_val_accuracy.append(val_accuracy_per_batch)
        # scheduler.step(val_loss)

        print(f"Epoch = [bold red]{epoch}/{epochs}[/bold red] | Train_loss = [bold yellow]{train_loss_per_batch:.5f} [/bold yellow] \
| Accuracy_train = [bold yellow]{train_accuracy_per_batch:.5f}[/bold yellow] \
| Val_loss = [bold green]{val_loss_per_batch:.5f}[/bold green] | Accuracy_val = [bold green]{val_accuracy_per_batch:.5f}[/bold green]", end="\r")

        if val_loss_per_batch < best_loss:
            best_loss = val_loss_per_batch
            model_path = Path(writer.log_dir, f'best_model.pt')
            torch.save(model.state_dict(), model_path)

        history = {"train_loss": all_train_loss,
                   "val_loss": all_val_loss,
                   "train_accuracy": all_train_accuracy,
                   "val_accuracy": all_val_accuracy
                   }
        if epoch % 5 == 0:
            plotTrainValidCurve(history, writer=writer)

    return history
