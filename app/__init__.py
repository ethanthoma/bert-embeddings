import download_nltk_data as dnd
import embed
import sentiment


def main():
    dnd.download()

    comments = "data/comments.csv"

    embed.get(comments, "./data/comments_with_embeddings.csv")
    sentiment.get(comments, "./data/sentiment.csv")


if __name__ == "__main__":
    main()
