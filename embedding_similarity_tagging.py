import os
import json
import torch
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from binary_classifier import BinaryClassifier

GND_TAGS_FILE = "shared-task-datasets/GND/dataset/GND-Subjects-tib-core.json"
TEST_DOCS_DIR = "shared-task-datasets/TIBKAT/tib-core-subjects/data/dev"  # Subfolders: train, dev, test
RESULTS_DIR = "results/dev"

MODEL_NAME = "distiluse-base-multilingual-cased-v1"  # Multilingual SentenceTransformer model
TAG_EMBEDDINGS_FILE = f"tag_embeddings_{MODEL_NAME}.json"
TOP_K = 50  # Number of top similar tags to retrieve for each document
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available


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
    Create a detailed textual description from a GND subject record using fields from the JSON schema.

    Args:
        tag (dict): A dictionary representing a GND subject.

    Returns:
        str: A structured textual description of the subject.
    """
    parts = []

    # Include the GND code
    if tag.get("Code") and tag["Code"].strip():
        parts.append(f"GND Code: {tag['Code']}.")

    # Include the classification details
    if tag.get("Classification Number") and tag["Classification Number"].strip():
        parts.append(f"Classification Number: {tag['Classification Number']}.")
    if tag.get("Classification Name") and tag["Classification Name"].strip():
        parts.append(f"Classification Name: {tag['Classification Name']}.")

    # Include the subject's main name
    if tag.get("Name") and tag["Name"].strip():
        parts.append(f"Subject Name: {tag['Name']}.")

    # Include alternate names if present
    if tag.get("Alternate Name") and isinstance(tag["Alternate Name"], list) and any(tag["Alternate Name"]):
        alternate_names = ", ".join(filter(None, tag["Alternate Name"]))
        if alternate_names:
            parts.append(f"Alternate Names: {alternate_names}.")

    # Include related subjects if present
    if tag.get("Related Subjects") and isinstance(tag["Related Subjects"], list) and any(tag["Related Subjects"]):
        related_subjects = ", ".join(filter(None, tag["Related Subjects"]))
        if related_subjects:
            parts.append(f"Related Subjects: {related_subjects}.")

    # Include source information if present
    if tag.get("Source") and tag["Source"].strip():
        parts.append(f"Source: {tag['Source']}.")
    if tag.get("Source URL") and tag["Source URL"].strip():
        parts.append(f"Source URL: {tag['Source URL']}.")

    # Include a formal definition if present
    if tag.get("Definition") and tag["Definition"].strip():
        parts.append(f"Definition: {tag['Definition']}.")

    return "\n".join(parts)  # Combine all parts into a single description string


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
    Extract a summarized text template from a JSON-LD document.

    Processes each item in the JSON-LD '@graph' list and extracts specific fields (e.g. title, abstract, description).
    For list values, the elements are joined using '. ' as a separator.
    Any value that looks like a link or contains an internal identifier (e.g., "gnd:") is skipped.

    Args:
        doc_json (dict): A dictionary representing a JSON-LD document.

    Returns:
        str: A formatted string with the extracted information.
    """

    # Mapping of JSON field names to human-readable labels
    fields = ["title", "abstract", "description", "subject", "creator", "contributor", "publisher", "issued"]

    def clean(text):
        """Strip and validate a text value, skipping links and internal IDs."""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text or text.startswith("http://") or text.startswith("https://") or "gnd:" in text:
            return None
        return text

    summary_lines = []

    # Process each item in the JSON-LD graph
    for item in doc_json.get("@graph", []):
        for field in fields:
            if field not in item:
                continue

            raw_value = item[field]
            if isinstance(raw_value, list):  # Clean and join list elements
                values = [clean(elem) for elem in raw_value]
                values = [v for v in values if v]
                if not values:
                    continue
                value = ", ".join(values)
            else:
                value = clean(raw_value)
                if not value:
                    continue

            # Add the field and value to the summary
            summary_lines.append(f"{field.capitalize()}: {value}")

    return "\n".join(summary_lines)


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


def tag_documents(model, tag_ids, tag_embeddings, test_docs_dir, top_k=TOP_K, results_dir=RESULTS_DIR, mlp_model='', doc_embeddings_filename=''):
    """
    Tag documents with the most similar GND tags.

    Args:
        model (SentenceTransformer): The SentenceTransformer model to use for encoding.
        tag_ids (list): List of tag IDs.
        tag_embeddings (Tensor): Embeddings of the tags.
        test_docs_dir (str): Directory containing the test documents.
        top_k (int, optional): Number of top similar tags to retrieve for each document. Defaults to TOP_K.
        results_dir (str, optional): Directory to save the tagging results. Defaults to RESULTS_DIR.
        mlp_model (str, optional): Path to the MLP model file for tagging. Defaults to ''. If provided, the MLP model will be used for tagging, otherwise cosine similarity will be used.

    Returns:
        dict: A dictionary where keys are document file paths and values are lists of top similar tags with their scores.
    """
    doc_files = get_all_doc_files(test_docs_dir)
    results = { }
    print(f"Found {len(doc_files)} documents in {test_docs_dir}.")

    doc_files_content = []

    # If the document embeddings don't exist already, compute them
    if not os.path.exists(doc_embeddings_filename):
        for doc_file in tqdm(doc_files, desc="Reading documents"):
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
            doc_files_content.append(doc_text)

        # Encode the document text
        print("Encoding documents...")
        doc_embeddings = model.encode(doc_files_content, convert_to_tensor=True, dtype=torch.float32, device=DEVICE, show_progress_bar=True)

        # Save the document embeddings
        os.makedirs(os.path.dirname(doc_embeddings_filename), exist_ok=True)
        torch.save(doc_embeddings, doc_embeddings_filename)
    else:
        doc_embeddings = torch.load(doc_embeddings_filename)

    # Load MLP model if provided
    if mlp_model:
        # Load the MLP model
        state_dict = torch.load(mlp_model, map_location=DEVICE)
        IN_FEATURES = tag_embeddings.shape[1] * 2
        hidden_dimensions = [2 * IN_FEATURES, 2048, 1024]
        mlp = BinaryClassifier(IN_FEATURES, hidden_dimensions).to(DEVICE)
        mlp.load_state_dict(state_dict)
        mlp.eval()

    for doc_file, doc_embedding in tqdm(list(zip(doc_files, doc_embeddings)), desc="Tagging documents"):

        if mlp_model == '':  # Use Cosine Similarity
            # Compute cosine similarity between the document and all tag embeddings
            cos_scores = util.cos_sim(doc_embedding, tag_embeddings)[0]

            # Get the top_k tags (indices)
            top_results = torch.topk(cos_scores, top_k).indices

            # Sort them by similarity score
            top_results = top_results[torch.argsort(-cos_scores[top_results])]

        else:  # Compute the MLP model output
            with torch.no_grad():
                mlp_input = torch.hstack((doc_embedding.repeat(tag_embeddings.shape[0], 1), tag_embeddings))
                scores = mlp(mlp_input)
            scores = scores.squeeze(1)

            # Get the top_k tags (indices)
            top_results = torch.topk(scores, top_k).indices

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
    parser.add_argument('--doc_embeddings_file', type=str, default=f"embeddings/{MODEL_NAME}", help='Path to the file to save/load document embeddings.')
    parser.add_argument('--mlp_model', type=str, default='', help='Path to the MLP model file for tagging.')
    args = parser.parse_args()

    # Create folders if they don't exist
    if os.path.dirname(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    if os.path.dirname(args.tag_embeddings_file):
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
    tag_documents(model, tag_ids, tag_embeddings, args.docs_path, args.top_k, args.results_dir, args.mlp_model, args.doc_embeddings_file)
    print("Tagging complete. Individual results saved in corresponding files.")
