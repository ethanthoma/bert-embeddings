import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    blob = TextBlob(text)
    result = []
    for word in blob.words:
        lemma_word = word.lemmatize().lower()
        if lemma_word not in stop_words:
            result.append(lemma_word)
    return " ".join(result)
