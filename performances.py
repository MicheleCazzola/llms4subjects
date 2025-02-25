import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def get_performance(model): 
    """
    ## Resource utilization
    ### Latency
    """


    STEPS = 10
    x = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer metus massa, aliquam vitae nibh a, pellentesque hendrerit tellus. Praesent interdum, magna quis gravida congue, leo dui dapibus tortor, eget consectetur magna quam eu libero. Vestibulum iaculis, mauris non pulvinar tempor, diam quam molestie erat, nec fermentum magna orci vitae nisi. Aliquam vestibulum non lectus ut lacinia. Nulla fermentum sapien eget pellentesque eleifend. Phasellus mauris neque, congue nec magna ac, dictum laoreet leo. Suspendisse potenti. Proin eu mauris sem. Pellentesque mattis fermentum dui congue mattis. Aenean leo sem, tempor ac malesuada a, placerat non dui. Ut vulputate venenatis sem eu tristique.
        Proin auctor mi libero, vel varius lectus consectetur eu. Aliquam porttitor, libero id posuere venenatis, lacus neque elementum nibh, et fringilla purus lacus vel quam. Pellentesque egestas dolor ligula, vitae eleifend urna finibus nec. Cras malesuada nibh urna. Pellentesque tincidunt urna quis turpis venenatis facilisis. Mauris sit amet ipsum mi. Aliquam sit amet ex id odio cursus gravida. Curabitur efficitur arcu molestie mauris varius, eget consequat nibh aliquam. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Quisque ac ipsum in velit efficitur pulvinar. Morbi condimentum tempor ultricies. In vel vulputate justo. Donec lobortis accumsan odio in molestie."""

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in tqdm(range(STEPS)):
        _ = model.encode(x)
    end_event.record()

    torch.cuda.synchronize()

    latency_ms = start_event.elapsed_time(end_event) / STEPS

    print("Latency (ms):", latency_ms)

    """### Params"""

    params = sum(p.numel() for p in model.parameters())
    print("Params:", params)

    """### FLOPs"""

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for model_name in [
        "sentence-transformers/all-MiniLM-L6-v2",
        "distiluse-base-multilingual-cased-v1",
        "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
        "intfloat/multilingual-e5-large"
    ]:
        model = SentenceTransformer(model_name)
        model.to(device)

        print(f"Model loaded: {model_name}")
        get_performance(model)
        print("-" * 50)

