import json
import torch
import random
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, SentencesDataset, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.trainer import WandbCallback
from embedding_similarity_tagging import load_gnd_tags, create_tag_description, get_all_doc_files, extract_text_from_jsonld


def extract_ground_truth_subjects(doc_json):
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


def load_gnd_mapping(gnd_tags_file):
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


def build_training_examples(training_docs_dir, gnd_mapping, all_tag_ids):
    """
    Build a list of InputExample objects for fine-tuning.
    For each training document:
      - The anchor is the document text (extracted via extract_text_from_jsonld).
      - For each ground truth subject, the positive is the corresponding GND tag description.
      - A negative is randomly selected from tags not in the ground truth.
    """
    examples = []
    doc_files = get_all_doc_files(training_docs_dir)
    print(f"Found {len(doc_files)} training documents in {training_docs_dir}.")

    for doc_file in tqdm(doc_files, desc="Building training examples"):
        try:
            with open(doc_file, "r", encoding="utf-8") as f:
                doc_json = json.load(f)
        except Exception as e:
            print(f"Error reading {doc_file}: {e}")
            continue

        anchor_text = extract_text_from_jsonld(doc_json)
        if not anchor_text:
            continue

        gt_subjects = extract_ground_truth_subjects(doc_json)
        if not gt_subjects:
            continue

        for subj in gt_subjects:
            if subj not in gnd_mapping:
                continue
            positive_text = gnd_mapping[subj]
            possible_negatives = [tid for tid in all_tag_ids if tid not in gt_subjects]
            if not possible_negatives:
                continue
            negative_id = random.choice(possible_negatives)
            negative_text = gnd_mapping.get(negative_id, "")
            if not negative_text:
                continue

            example = InputExample(texts=[anchor_text, positive_text, negative_text])
            examples.append(example)
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a SentenceTransformer on training data for subject tagging")
    parser.add_argument('--training_path', type=str, required=True, help="Path to the training data folder (JSON-LD documents)")
    parser.add_argument('--gnd_tags_file', type=str, default="shared-task-datasets/GND/dataset/GND-Subjects-tib-core.json", help="Path to the GND tags JSON file")
    parser.add_argument('--model_name', type=str, default="distiluse-base-multilingual-cased-v1", help="Pretrained SentenceTransformer model name")
    parser.add_argument('--output_model_path', type=str, default="finetuned_model", help="Path to save the fine-tuned model")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of training epochs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")
    model = SentenceTransformer(args.model_name, device=device)

    print("Loading GND tags and building mapping...")
    gnd_mapping, all_tag_ids = load_gnd_mapping(args.gnd_tags_file)
    print(f"Loaded {len(gnd_mapping)} GND tags.")

    print("Building training examples...")
    examples = build_training_examples(args.training_path, gnd_mapping, all_tag_ids)
    print(f"Created {len(examples)} training examples.")
    if not examples:
        print("No training examples found. Exiting.")
        exit(1)

    # Convert training examples into a SentencesDataset
    train_dataset = SentencesDataset(examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    # Define the training loss (using TripletLoss here)
    train_loss = losses.TripletLoss(model)

    # Set up training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_model_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        warmup_steps=int(0.1 * len(train_dataloader)),
        overwrite_output_dir=True,

        # Optional tracking/debugging parameters
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=4,
        logging_steps=100,
        load_best_model_at_end=True
    )

    # Create a Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        callbacks=[WandbCallback()]  # Init WandB callback for logging
    )

    print("Starting fine-tuning...")
    trainer.train()
    model.save(args.output_model_path)
    print(f"Fine-tuning complete. Model saved to {args.output_model_path}.")
