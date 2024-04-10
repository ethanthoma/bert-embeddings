from . import embed
from . import sentiment


def main():
    input_csv_path = 'data/comments.csv'

    embed.get(input_csv_path, './data/embeddings.csv')

    sentiment.get(input_csv_path, 'data/sentiment.csv')


if __name__ == "__main__":
    main()
