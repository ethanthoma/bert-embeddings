from . import download_nltk_data as dnd
from . import embed

import os


def main():
    if not os.path.exists('./data'):
        os.mkdir('./data')

    dnd.download()
    embed.embed_and_save('./data/comments.csv',
                         './data/comments_with_embeddings.csv')


if __name__ == '__main__':
    main()
