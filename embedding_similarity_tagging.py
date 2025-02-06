import os
import json
import torch
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

GND_TAGS_FILE = "shared-task-datasets/GND/dataset/GND-Subjects-tib-core.json"
TEST_DOCS_DIR = "shared-task-datasets/TIBKAT/tib-core-subjects/data/dev"  # Subfolders: train, dev, test
RESULTS_DIR = "bert_similarity_tagging_results/dev"

MODEL_NAME = "distiluse-base-multilingual-cased-v1"  # Multilingual SentenceTransformer model
TAG_EMBEDDINGS_FILE = f"tag_embeddings_{MODEL_NAME}.json"
TOP_K = 50  # Number of top similar tags to retrieve for each document
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available


def load_gnd_tags(gnd_file):
    """
    Load GND tags from a JSON file.

    Args:
        gnd_file (str): Path to the GND tags JSON file.

    Returns:
        list: List of tags loaded from the JSON file.
    """
    with open(gnd_file, "r", encoding="utf-8") as f:
        tags = json.load(f)
    return tags


def create_tag_description(tag):
    """
    Create a textual description from a tag using fields from the JSON schema.

    Args:
        tag (dict): A dictionary representing a tag.

    Returns:
        str: A textual description of the tag.
    """
    parts = []
    if "preferredName" in tag:
        parts.append(f"Name: {tag['preferredName']}.")
    if "scopeNote" in tag:
        parts.append(f"Description: {tag['scopeNote']}.")
    if "broader" in tag and tag["broader"]:
        broader_terms = ", ".join(tag["broader"])
        parts.append(f"Broader terms: {broader_terms}.")
    return " ".join(parts)


def prepare_tag_embeddings(model, tags):
    """
    Prepare embeddings for the GND tags.

    Args:
        model (SentenceTransformer): The SentenceTransformer model to use for encoding.
        tags (list): List of tags to encode.

    Returns:
        tuple: A tuple containing tag IDs, tag texts, and their embeddings.
    """

    if os.path.exists(TAG_EMBEDDINGS_FILE):
        with open(TAG_EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["tag_ids"], data["tag_texts"], torch.tensor(data["tag_embeddings"], dtype=torch.float32, device=DEVICE)

    tag_texts = []
    tag_ids = []
    for tag in tags:
        text = create_tag_description(tag)
        tag_texts.append(text)  # Add the tag description to the list
        tag_ids.append(tag.get("Code", tag.get("Name", "unknown")))  # Add the tag ID to the list

    print("Encoding tag descriptions...")
    tag_embeddings = model.encode(tag_texts, convert_to_tensor=True, show_progress_bar=True, dtype=torch.float32, device=DEVICE)

    with open(TAG_EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump({ "tag_ids": tag_ids, "tag_texts": tag_texts, "tag_embeddings": tag_embeddings.tolist() }, f, ensure_ascii=False, indent=2)

    return tag_ids, tag_texts, tag_embeddings


def extract_text_from_jsonld(doc_json):
    """
    Extract a text string from a JSON-LD document.

    Args:
        doc_json (dict): A dictionary representing a JSON-LD document.

    Returns:
        str: Extracted text from the document.
    """
    parts = []
    for item in doc_json.get("@graph", []):
        fields = ["title", "abstract", "description"]

        for field in fields:
            if field in item:
                value = item[field]
                # If the attribute is a list, join the elements.
                # This may happen for example if Abstract is multiline
                if isinstance(value, list):
                    value = " ".join(value)
                parts.append(value)

    return " ".join(parts).strip()


def get_all_doc_files(root_dir):
    """
    Recursively collect all JSON(-LD) files in the directory.

    Args:
        root_dir (str): Root directory to search for JSON(-LD) files.

    Returns:
        list: List of file paths to JSON(-LD) files.
    """
    doc_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".json") or file.lower().endswith(".jsonld"):
                doc_files.append(os.path.join(subdir, file))
    return doc_files


def tag_documents(model, tag_ids, tag_embeddings, test_docs_dir, top_k=TOP_K, results_dir=RESULTS_DIR):
    """
    Tag documents with the most similar GND tags.

    Args:
        model (SentenceTransformer): The SentenceTransformer model to use for encoding.
        tag_ids (list): List of tag IDs.
        tag_embeddings (Tensor): Embeddings of the tags.
        test_docs_dir (str): Directory containing the test documents.
        top_k (int, optional): Number of top similar tags to retrieve for each document. Defaults to TOP_K.
        results_dir (str, optional): Directory to save the tagging results. Defaults to RESULTS_DIR.

    Returns:
        dict: A dictionary where keys are document file paths and values are lists of top similar tags with their scores.
    """
    doc_files = get_all_doc_files(test_docs_dir)
    results = { }
    print(f"Found {len(doc_files)} documents in {test_docs_dir}.")

    for doc_file in tqdm(doc_files, desc="Tagging documents"):
        try:
            with open(doc_file, "r", encoding="utf-8") as f:
                doc_json = json.load(f)
        except Exception as e:
            print(f"Error reading {doc_file}: {e}")
            continue

        doc_text = extract_text_from_jsonld(doc_json)
        if not doc_text:
            print(f"No text found in {doc_file}. Skipping.")
            continue

        # Encode the document text
        doc_embedding = model.encode(doc_text, convert_to_tensor=True, dtype=torch.float32, device=DEVICE)

        # Compute cosine similarity between the document and all tag embeddings
        cos_scores = util.cos_sim(doc_embedding, tag_embeddings)[0]

        # Get the top_k tags (indices)
        top_results = torch.topk(cos_scores, top_k).indices

        # Sort them by similarity score
        top_results = top_results[torch.argsort(-cos_scores[top_results])]

        # Store the results
        doc_result = { "dcterms:subject": [tag_ids[idx] for idx in top_results] }

        # Save the results in a new JSON file, in the subfolder <RESULTS_DIR>/<original path>/<original filename>.json
        output_filename = os.path.join(results_dir, os.path.relpath(os.path.dirname(doc_file), test_docs_dir), os.path.splitext(os.path.basename(doc_file))[0] + ".json")
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(doc_result, f, ensure_ascii=False, indent=2)

    return results


# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMs4Subjects Solution: Tagging documents via embeddings & cosine similarity')
    parser.add_argument('--tags_file', type=str, default=GND_TAGS_FILE, help='Path to the JSON file containing GND tags to use for tagging.')
    parser.add_argument('--docs_path', type=str, default=TEST_DOCS_DIR, help='Path to the directory containing the JSON-LD documents.')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='Name of the SentenceTransformer model to use.')
    parser.add_argument('--top_k', type=int, default=TOP_K, help='Number of top similar tags to retrieve for each document.')
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR, help='Directory to save the tagging results.')
    parser.add_argument('--tag_embeddings_file', type=str, default=TAG_EMBEDDINGS_FILE, help='Path to the file to save/load tag embeddings.')
    args = parser.parse_args()

    # Create folders if they don't exist
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.tag_embeddings_file), exist_ok=True)

    # Load the SentenceTransformer model
    print("Loading model...")
    model = SentenceTransformer(args.model_name)
    model.to(DEVICE)

    # Load and process the GND tags
    print("Loading GND tags...")
    TAG_EMBEDDINGS_FILE = args.tag_embeddings_file
    tags = load_gnd_tags(args.tags_file)
    tag_ids, tag_texts, tag_embeddings = prepare_tag_embeddings(model, tags)

    # Process documents, compute similarities, and save results
    print("Processing test documents and computing similarities...")
    tag_documents(model, tag_ids, tag_embeddings, args.docs_path, args.top_k, args.results_dir)
    print("Tagging complete. Individual results saved in corresponding files.")
