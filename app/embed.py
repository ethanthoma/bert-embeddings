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
        # use the pooled output as sentence-level embeddings
        embeddings = outputs.pooler_output
    return embeddings


def embed_and_save(filepath, savepath):
    df_comments = pd.read_csv(filepath)
    text_embeddings = encode_texts(df_comments['text'].tolist())

    embeddings_df = pd.DataFrame(text_embeddings)

    embeddings_df.columns = [
        f'embedding_{i}' for i in range(embeddings_df.shape[1])]

    df_final = pd.concat(
        [df_comments.drop(columns=['text']), embeddings_df], axis=1)

    df_final.to_csv(savepath, index=False)
