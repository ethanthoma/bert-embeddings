import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from .preprocess import preprocess_text

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", max_length=512)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", max_length=512
)
model.to(device)

sentiment_pipeline = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer, device=device
)


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


def analyze_sentiment(token_segments):
    sentiments = []
    for segment in token_segments:
        segment_text = tokenizer.decode(segment, skip_special_tokens=True)
        sentiment = sentiment_pipeline(segment_text)[0]
        sentiments.append(sentiment)
    return sentiments


def combine_sentiments(sentiments):
    avg_score = sum([sentiment["score"]
                    for sentiment in sentiments]) / len(sentiments)
    overall_sentiment = "POSITIVE" if avg_score >= 0.5 else "NEGATIVE"
    return overall_sentiment, avg_score


def process_chunk(chunk):
    results = []
    for text in chunk["text"]:
        preprocessed_text = preprocess_text(text)
        text_segments = segment_text(preprocessed_text)
        sentiments = analyze_sentiment(text_segments)
        overall_sentiment, avg_score = combine_sentiments(sentiments)
        results.append((overall_sentiment, avg_score))

    sentiments, scores = zip(*results)
    chunk["sentiment"] = sentiments
    chunk["sentiment_score"] = scores
    return chunk


def get(input_csv_path, output_csv_path, chunk_size=100):
    print("Loading...")

    for i, chunk in enumerate(pd.read_csv(input_csv_path, chunksize=chunk_size)):
        processed_chunk = process_chunk(chunk)
        processed_chunk.to_csv(output_csv_path, mode="a",
                               index=False, header=i == 0)
        print(f"Processed chunk {i + 1}.")

    print("Sentiments complete.")
