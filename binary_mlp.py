import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sentence_transformers import SentenceTransformer
from datasets import Dataset
import os
from finetune_sentence_transformer import load_gnd_mapping, generate_dataset, save_dataset, load_dataset
import argparse
import matplotlib.pyplot as plt
from binary_classifier import BinaryClassifier


def get_embeddings(embedding_file: str, encoder: SentenceTransformer, data, device) -> Dataset:
    if os.path.exists(embedding_file):
        embedding_dataset = torch.load(embedding_file, weights_only=False)
    else:
        anchors = encoder.encode(data["anchor"], convert_to_tensor=True, dtype=torch.float32, device=device, show_progress_bar=True)
        print("Embedded anchors")
        positives = encoder.encode(data["positive"], convert_to_tensor=True, dtype=torch.float32, device=device, show_progress_bar=True)
        print("Embedded positives")
        negatives = encoder.encode(data["negative"], convert_to_tensor=True, dtype=torch.float32, device=device, show_progress_bar=True)
        print("Embedded negatives")
        embedding_dataset = Dataset.from_dict({
            "sentence1": torch.cat([anchors, anchors]),
            "sentence2": torch.cat([positives, negatives]),
            "score": [1.0] * len(positives) + [0.0] * len(negatives)
        })

        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
        torch.save(embedding_dataset, embedding_file)
    return embedding_dataset


def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device: torch.device, log_frequency: int):
    """ Evaluate the model on the given dataset
    :param model: Model to evaluate
    :param dataloader: DataLoader for the evaluation dataset
    :param criterion: Loss function
    :param device: Device to run the evaluation on ('cuda' or 'cpu')
    :param log_frequency: Frequency to log the evaluation loss
    :return: Average loss on the evaluation dataset
    """
    model.eval()
    steps = 0
    eval_loss = 0

    for batch in dataloader:
        text = torch.vstack(batch["sentence1"]).permute(1, 0).float()
        tag = torch.vstack(batch["sentence2"]).permute(1, 0).float()
        label = batch["score"].unsqueeze(1).float()

        input = torch.hstack((text, tag)).to(device)
        label[label < 0] = 0
        label = label.to(device)

        pred = model(input)
        loss = criterion(pred, label)

        eval_loss += loss.item()

        if (steps + 1) % log_frequency == 0:
            print(f"\tValidation: Iteration {steps + 1}, Loss: {eval_loss / (steps + 1)}")

        steps += 1

    return eval_loss / len(dataloader)


def train(
        model: nn.Module,
        trainloader: DataLoader,
        validloader: DataLoader,
        num_epochs: int,
        criterion,
        optimizer,
        device: torch.device,
        train_log_frequency: int,
        val_log_frequency: int,
        eval_epoch_frequency: int,
        encoder_name: str,
        scheduler=None
):
    """ Train the model """
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        steps = 0
        train_loss = 0
        for batch in trainloader:
            optimizer.zero_grad()
            text = torch.vstack(batch["sentence1"]).permute(1, 0).float()
            tag = torch.vstack(batch["sentence2"]).permute(1, 0).float()
            label = batch["score"].unsqueeze(1).float()

            input = torch.hstack((text, tag)).to(device)
            label[label < 0] = 0
            label = label.to(device)

            pred = model(input)
            loss = criterion(pred, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (steps + 1) % train_log_frequency == 0:
                print(f"Epoch {epoch + 1}, Iteration {steps + 1}, Loss: {train_loss / (steps + 1)}")

            steps += 1

        train_loss /= len(trainloader)
        train_losses.append(train_loss)

        if (epoch + 1) % eval_epoch_frequency == 0:
            val_loss = evaluate(model, validloader, criterion, device, val_log_frequency)
            val_losses.append(val_loss)

        if (epoch + 1) % eval_epoch_frequency == 0:
            print(f"End of epoch {epoch + 1}, Training loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}" + (f", LR: {scheduler.get_last_lr()[0]}" if scheduler else ""))
        else:
            print(f"End of epoch {epoch + 1}, Training loss: {train_losses[-1]}")
        print("--------------------------------------------------")

        if val_loss <= min(val_losses):
            os.makedirs("models/mlp", exist_ok=True)
            torch.save(model.state_dict(), f"models/mlp/{encoder_name}.pth")

        if scheduler is not None:
            scheduler.step()

    return train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary MLP")
    parser.add_argument('--model_name', type=str, default="distiluse-base-multilingual-cased-v1", help="Pretrained SentenceTransformer model name")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--gnd_tags_file', type=str, default="shared-task-datasets/GND/dataset/GND-Subjects-tib-core.json", help="Path to the GND tags JSON file")
    parser.add_argument('--training_path', type=str, default="shared-task-datasets/GND/dataset/train", help="Path to the training dataset")
    parser.add_argument('--eval_path', type=str, default="shared-task-datasets/GND/dataset/eval", help="Path to the evaluation dataset")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading model...")
    encoder = SentenceTransformer(args.model_name)
    encoder.to(device)
    print("Model loaded.")

    print("Loading GND tags and building mapping...")
    gnd_mapping, all_tag_ids = load_gnd_mapping(args.gnd_tags_file)
    print(f"Loaded {len(gnd_mapping)} GND tags.")

    # ---- Build/Load examples for training ----
    print("Building training examples...")
    train_dataset_file = "train_dataset.json"
    if os.path.exists(train_dataset_file):
        train_dataset = load_dataset(train_dataset_file)
    else:
        train_dataset = generate_dataset(args.training_path, gnd_mapping, all_tag_ids)
        save_dataset(train_dataset, train_dataset_file)
    print(f"Created {len(train_dataset)} training examples.")

    # ---- Build/Load examples for evaluation ----
    print("Building evaluation examples...")
    eval_dataset_file = "eval_dataset.json"
    if os.path.exists(eval_dataset_file):
        eval_dataset = load_dataset(eval_dataset_file)
    else:
        eval_dataset = generate_dataset(args.eval_path, gnd_mapping, all_tag_ids)
        save_dataset(eval_dataset, eval_dataset_file)
    print(f"Created {len(eval_dataset)} evaluation examples.")

    # ---- Get embeddings ----
    print("Getting embeddings...")
    train_dataset_embedding = get_embeddings(f"embeddings/{args.model_name}_train_embeddings.pt", encoder, train_dataset, device)
    eval_dataset_embedding = get_embeddings(f"embeddings/{args.model_name}_eval_embeddings.pt", encoder, eval_dataset, device)
    print("Embeddings obtained.")

    # ---- Adding a MLP on top of the embeddings ----
    EMBEDDING_DIM = encoder.get_sentence_embedding_dimension()
    IN_FEATURES = EMBEDDING_DIM * 2

    hidden_dimensions = [2 * IN_FEATURES, 2048, 1024]
    model = BinaryClassifier(IN_FEATURES, hidden_dimensions)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    model.to(device)

    trainloader = DataLoader(train_dataset_embedding, batch_size=args.batch_size, shuffle=True)
    validloader = DataLoader(eval_dataset_embedding, batch_size=args.batch_size, shuffle=False)

    train_losses, validation_losses = train(
        model,
        trainloader,
        validloader,
        args.num_epochs,
        criterion,
        optimizer,
        device,
        train_log_frequency=2000,
        val_log_frequency=500,
        eval_epoch_frequency=1,
        encoder_name=args.model_name,
        # scheduler=scheduler
    )

    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xticks(range(0, args.num_epochs), range(1, args.num_epochs + 1))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    '''
    """## Resource utilization

    ### Latency
    """

    STEPS = 100_000
    x = torch.randn((1, IN_FEATURES)).to(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(STEPS):
        _ = model(x)
    end_event.record()

    torch.cuda.synchronize()

    latency_ms = start_event.elapsed_time(end_event) / STEPS

    print(latency_ms)

    """### Params"""

    num_params = sum([p.numel() for p in model.parameters()])
    print(num_params)

    """### FLOPS"""

    from fvcore.nn import FlopCountAnalysis
    import wandb

    flops = FlopCountAnalysis(model, x)
    print(flops.total())'''
