import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from .preprocess import preprocess_text


device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.to(device)


def segment_text(text, max_length=510):
    tokens = tokenizer.encode(
        text, add_special_tokens=True, truncation=True, max_length=max_length
    )

    segments = []
    for i in range(0, len(tokens), max_length):
        segment_ids = tokens[i: i + max_length]
        if len(segment_ids) > max_length:
            segment_ids = segment_ids[:max_length]
        segments.append(segment_ids)

    return segments


def get_embeddings(token_segments):
    embeddings = []
    for segment_ids in token_segments:
        input_ids = torch.tensor(segment_ids).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embedding)
    return embeddings


def combine_embeddings(embeddings):
    avg_embedding = torch.mean(
        torch.stack([torch.Tensor(embed) for embed in embeddings]), dim=0
    ).numpy()
    return avg_embedding


def process_chunk(chunk):
    embeddings_list = []
    for text in chunk["text"]:
        preprocessed_text = preprocess_text(text)
        text_segments = segment_text(preprocessed_text)
        embeddings = get_embeddings(text_segments)
        combined_embedding = combine_embeddings(embeddings)
        embeddings_list.append(combined_embedding)

    embeddings_df = pd.DataFrame([embedding.flatten()
                                 for embedding in embeddings_list])
    return pd.concat([chunk.reset_index(drop=True), embeddings_df], axis=1)


def get(input_csv_path, output_csv_path, chunk_size=100):
    print("Loading...")

    for i, chunk in enumerate(pd.read_csv(input_csv_path, chunksize=chunk_size)):
        processed_chunk = process_chunk(chunk)
        processed_chunk.to_csv(output_csv_path, mode="a",
                               index=False, header=i == 0)
        print(f"Processed chunk {i + 1}.")

    print("Embeddings extraction complete.")
