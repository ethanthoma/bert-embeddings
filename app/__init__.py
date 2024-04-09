from . import embed
from . import sentiment
import pandas as pd


def main():
    input_csv_path = 'data/comments.csv'

    print(pd.read_csv('data/sentiment.csv').shape)

    exit()

    embed.get(input_csv_path, './data/comments_with_embeddings.csv')

    sentiment.get(input_csv_path, 'data/sentiment.csv')


if __name__ == "__main__":
    main()
