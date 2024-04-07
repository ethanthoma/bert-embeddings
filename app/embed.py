import pandas as pd
import torch
from textblob import TextBlob
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def preprocess_text(text):
    blob = TextBlob(text)
    lemmatized_text = ' '.join([word.lemmatize() for word in blob.words])
    return lemmatized_text


def encode_texts(texts):
    processed_texts = [preprocess_text(text) for text in texts]
    encoded_inputs = tokenizer(
        processed_texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        embeddings = outputs.pooler_output
    return embeddings


def embed_and_save(filepath, savepath, chunk_size=100):
    reader = pd.read_csv(filepath, chunksize=chunk_size)

    first_chunk = True
    for i, df_chunk in enumerate(reader):
        text_embeddings = encode_texts(df_chunk['text'].tolist())
        embeddings_df = pd.DataFrame(text_embeddings.numpy())
        embeddings_df.columns = [
            f'embedding_{i}' for i in range(embeddings_df.shape[1])]
        df_chunk_processed = pd.concat(
            [df_chunk.drop(columns=['text']), embeddings_df], axis=1)

        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        df_chunk_processed.to_csv(
            savepath, mode=mode, header=header, index=False)

        if first_chunk:
            first_chunk = False

        print(f'Chunk {i+1} processed')
