import sys
import string
import nltk
from nltk.corpus import stopwords
from numpy.lib.function_base import vectorize
from sklearn.feature_extraction.text import TfidfVectorizer

#converted accent characters
def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text

#expand contraction
def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = list(cont.expand_texts([text], precise=True))[0]
    return text


def preprocess(input:str):
    text_input = input.lower()
    text_input = text_input.replace('[^\w\s]', '')
    text_input_tokenized = nltk.word_tokenize(text_input)

    stop_words = stopwords.words('english')
    text_input_tokenized = [item for item in text_input_tokenized if item not in stop_words]
    return text_input_tokenized