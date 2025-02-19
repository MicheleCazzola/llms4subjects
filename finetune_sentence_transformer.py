import json
import os
import json
import wandb
import torch
import random
import argparse
from tqdm import tqdm
from datasets import Dataset, IterableDataset  # import of IterableDataset is needed for SentenceTransformer training
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.trainer import WandbCallback
from embedding_similarity_tagging import load_gnd_tags, create_tag_description, get_all_doc_files, extract_text_from_jsonld
import torch.nn as nn
from transformers import AutoModel
import torch.nn.utils.prune as prune


def extract_ground_truth_subjects(doc_json: str) -> list:
    """
    Extract ground truth subjects from a training JSON-LD document.
    Ground truth subjects are stored under the '@graph' list in a field named "dcterms:subject".
    Returns a list of subject codes.
    """
    subjects = []
    for item in doc_json.get("@graph", []):
        if "dcterms:subject" in item:
            subj = item["dcterms:subject"]
            if isinstance(subj, list):
                subjects.extend(subj)
            else:
                subjects.append(subj)

    subjects = [subject["@id"] for subject in subjects if "@id" in subject]  # extract GND codes
    return list(set(subjects))  # return unique subjects


def load_gnd_mapping(gnd_tags_file: str) -> tuple:
    """
    Build a mapping from GND tag codes to their detailed textual descriptions.
    Also returns a list of all tag codes.
    """
    tags = load_gnd_tags(gnd_tags_file)
    mapping = { }
    tag_ids = []
    for tag in tags:
        code = tag.get("Code", tag.get("Name", None))
        if code:
            mapping[code] = create_tag_description(tag)
            tag_ids.append(code)
    return mapping, tag_ids


def generate_dataset(docs_dir: str, gnd_mapping: dict, all_tag_ids: list) -> Dataset:
    """ Generate a Dataset object for fine-tuning, based on the data in the specified directory.
    dataset = Dataset.from_dict({ "anchor": anchors, "positive": positives, "negative": negatives })

    For each training document:
      - The anchor is the document text (extracted via extract_text_from_jsonld).
      - For each ground truth subject, the positive is the corresponding GND tag description.
      - A negative is randomly selected from tags not in the ground truth.

    Args:
        docs_dir (str): Directory containing the JSON-LD documents.
        gnd_mapping (dict): Mapping from GND tag codes to their detailed textual descriptions.
        all_tag_ids (list): List of all tag codes.
    """

    items = []
    doc_files = get_all_doc_files(docs_dir)  # Get all document files in the directory
    print(f"Found {len(doc_files)} documents in {docs_dir}.")

    for doc_file in tqdm(doc_files, desc="Building examples"):
        try:
            with open(doc_file, "r", encoding="utf-8") as f:
                doc_json = json.load(f)  # Load the JSON content of the document
        except Exception as e:
            print(f"Error reading {doc_file}: {e}")
            continue

        # Extract the main text from the document
        anchor_text = extract_text_from_jsonld(doc_json)
        if not anchor_text:
            continue

        # Extract ground truth subjects
        gt_subjects = extract_ground_truth_subjects(doc_json)
        if not gt_subjects:
            continue

        # For each ground truth subject, create a positive example and a negative example
        for subj in gt_subjects:
            if subj not in gnd_mapping:  # Skip if the GND tag is not in the mapping
                continue

            # Positive example
            positive_text = gnd_mapping[subj]  # Get the positive example text

            # Negative example
            possible_negatives = [tid for tid in all_tag_ids if tid not in gt_subjects]
            if not possible_negatives:
                continue

            negative_id = random.choice(possible_negatives)  # Randomly select a negative example
            negative_text = gnd_mapping.get(negative_id, "")
            if not negative_text:
                continue

            # Create an item with anchor, positive, and negative texts
            items.append({ "anchor": anchor_text, "positive": positive_text, "negative": negative_text })

    if len(items) == 0:
        print("Unable to generate dataset. Exiting.")
        exit(1)

    # Create a Dataset object from the items
    return Dataset.from_dict({
        "anchor": [item["anchor"] for item in items],
        "positive": [item["positive"] for item in items],
        "negative": [item["negative"] for item in items]
    })


def save_dataset(dataset: Dataset, output_file: str):
    """ Save the dataset to a JSON file """
    # Avoid the error: TypeError: Object of type Dataset is not JSON serializable
    with open(output_file, "w") as f:
        json.dump(dataset.to_dict(), f)


def load_dataset(input_file: str) -> Dataset:
    """ Load a dataset from a JSON file """
    with open(input_file, "r") as f:
        return Dataset.from_dict(json.load(f))


# From https://github.com/huggingface/peft/issues/41#issuecomment-1404611868
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param:.2f}%"
    )


class LoRA(nn.Module):
    def __init__(self, original_layer, rank=8):
        super(LoRA, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Initialize the Low-rank matrices A and B
        self.A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.B = nn.Parameter(torch.randn(size=(rank, self.out_features)))

    def forward(self, x):
        # The output is the original layer output plus the low-rank adaptation

        # LoRA output
        lora_output = torch.matmul(x, self.A)
        lora_output = torch.matmul(lora_output, self.B)

        # layer output, which combines the original output with the LoRA one
        return self.original_layer(x) + lora_output


def apply_lora(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRA(child))
        else:
            apply_lora(child)


def get_loss(loss_name: str, train_data: Dataset, eval_data: Dataset) -> tuple:
    """ Get the loss function and update the datasets if necessary.
    See https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html for details about losses

    Args:
        loss_name (str): Name of the loss function to use
        train_data (Dataset): Training dataset containing anchor, positive, and negative examples
        eval_data (Dataset): Evaluation dataset containing anchor, positive, and negative examples

    Returns:
        tuple: A tuple with loss function, updated training dataset, and updated evaluation dataset
    """

    if loss_name == "Triplet":  # Works with triplets of anchor, positive, negative
        loss = losses.TripletLoss(model=model)
    elif loss_name == "MultipleNegativesRanking":  # Works with pairs of anchor, positive
        loss = losses.MultipleNegativesRankingLoss(model=model)
        train_data = train_data.remove_columns("negative")
        eval_data = eval_data.remove_columns("negative")

    elif loss_name in ["CosineSimilarity", "coSENT", "AnglE"]:  # Works with 2 sentences as texts, and a "score" corresponding to the expected similarity

        if loss_name == "CosineSimilarity":
            loss = losses.CosineSimilarityLoss(model=model)
        elif loss_name == "coSENT":
            loss = losses.CoSENTLoss(model=model)
        elif loss_name == "AnglE":
            loss = losses.AnglELoss(model=model)

        anchors = train_data["anchor"]
        positives = train_data["positive"]
        negatives = train_data["negative"]
        train_data = Dataset.from_dict({ "sentence1": anchors + anchors, "sentence2": positives + negatives, "score": [1.0] * len(anchors) + [0.0] * len(negatives) })

        anchors = eval_data["anchor"]
        positives = eval_data["positive"]
        negatives = eval_data["negative"]
        eval_data = Dataset.from_dict({ "sentence1": anchors + anchors, "sentence2": positives + negatives, "score": [1.0] * len(anchors) + [0.0] * len(negatives) })

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

    return loss, train_data, eval_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a SentenceTransformer on training data for subject tagging")
    parser.add_argument('--training_path', type=str, required=True, help="Path to the training data folder (JSON-LD documents)")
    parser.add_argument('--eval_path', type=str, required=True, help="Path to the evaluation data folder (JSON-LD documents)")
    parser.add_argument('--gnd_tags_file', type=str, default="shared-task-datasets/GND/dataset/GND-Subjects-tib-core.json", help="Path to the GND tags JSON file")
    parser.add_argument('--model_name', type=str, default="distiluse-base-multilingual-cased-v1", help="Pretrained SentenceTransformer model name")
    parser.add_argument('--output_model_path', type=str, default="finetuned_model", help="Path to save the fine-tuned model")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--PEFT', type=int, default=0, help="Apply PEFT to the model (0: None, 1: LoRA, 2: BitFit)")
    parser.add_argument('--loss', type=str, default="Triplet", help="Loss function to use", choices=["Triplet", "MultipleNegativesRanking", "CosineSimilarity", "coSENT", "AnglE"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading model...")
    model = SentenceTransformer(args.model_name, device=device)

    if args.PEFT != 0:
        print("Model before PEFT:")
        print_trainable_parameters(model)
        if args.PEFT == 1:
            # Apply LoRA to all layers in the model
            apply_lora(model)
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze only LoRA parameters
            for layer in model.modules():
                if isinstance(layer, LoRA):
                    layer.A.requires_grad = True
                    layer.B.requires_grad = True
        elif args.PEFT == 2:
            # Apply BitFit to all layers in the model
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.requires_grad = True  # Unfreeze bias terms
                else:
                    param.requires_grad = False  # Freeze all other parameters
        else:
            print("Invalid PEFT option. Exiting.")
            exit(1)
        print("Model after PEFT:")
        print_trainable_parameters(model)

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

    # ---- Set up the loss function ----
    train_loss, train_dataset, eval_dataset = get_loss(args.loss, train_dataset, eval_dataset)

    # ---- Create the data loader ----
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    # ---- Set up training arguments ----
    run_name = f"finetune_{args.model_name.split('/')[-1]}_{args.loss}Loss"

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_model_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        warmup_steps=int(0.1 * len(train_dataloader)),
        overwrite_output_dir=True,

        # Optional tracking/debugging parameters
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=4,
        logging_steps=1000,
        run_name=run_name,  # Name of the WandB run
        load_best_model_at_end=True,
        optim_target_modules=[".*bias.*"] if args.PEFT == 2 else ["query", "key", "value", "dense"],
    )

    # ---- Create a Trainer and run training ----
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        # callbacks=[WandbCallback()]  # Init WandB callback for logging
    )

    print("Starting fine-tuning...")
    trainer.train()
    model.save(args.output_model_path)
    print(f"Fine-tuning complete. Model saved to {args.output_model_path}.")
    wandb.finish()  # Finish the WandB run
