import torch
from src.NN import CustomDataset, DataLoader
from src.Metrics import binary_acc
from datetime import datetime
from numpy import inf
from rich.console import Console
from rich import print


def train_and_validation(model,
                         train_dataset: CustomDataset,
                         val_dataset: CustomDataset,
                         batch_size: int,
                         epochs: int,
                         device: str,
                         criterion: torch.nn.modules.loss,
                         optimizer: torch.optim.Adam,
                         scheduler=None):
    console = Console()
    all_train_loss = []
    all_val_loss = []
    all_train_accuracy = []
    all_val_accuracy = []
    best_loss = inf

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')

    print("INFO: start training...")
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
        all_val_accuracy.append(val_acc / val_accuracy_per_batch)
        # scheduler.step(val_loss)

        # console.print(f"[bold purple] | Loss = [/bold purple]{(final_loss):.8f} [bold purple]| Accuracy = [/bold purple] \
        # [bold green]{(sum_acc / count_acc):.8f}[/bold green]", end="\r")
        #console.print("\r", f"[bold white] epoch = {epoch}/{epochs} [/bold white]", end="")
        print(f"Epoch = [bold red]{epoch}/{epochs}[/bold red] | Train_loss = [bold yellow]{train_loss_per_batch:.5f} [/bold yellow] \
| Accuracy_train = [bold yellow]{train_accuracy_per_batch:.5f}[/bold yellow] \
| Val_loss = [bold green]{val_loss_per_batch:.5f}[/bold green] | Accuracy_val = [bold green]{val_accuracy_per_batch:.5f}[/bold green]", end="\r")
        # print("Val loss: ", val_loss_per_batch)
        # print("best loss: ", val_accuracy_per_batch)
        if val_loss_per_batch < best_loss:
            best_loss = val_loss_per_batch
            # print(f"Best model Accuracy: {val_loss_per_batch} | Loss {val_accuracy_per_batch}")
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), "best_model.pt")

    history = {}
    history["train_loss"] = all_train_loss
    history["val_loss"] = all_val_loss
    history["train_accuracy"] = all_train_accuracy
    history["val_accuracy"] = all_val_accuracy

    return history