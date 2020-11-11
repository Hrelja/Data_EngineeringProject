import nltk
from nltk.corpus import stopwords
import unidecode
import string
import contractions as cont
import numpy as np

#converted accent characters
def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text

def expand_contractions(text):
    return cont.fix(text)

def del_punctuations(text):
    return text.translate(
        str.maketrans('', '', string.punctuation))

def preprocess(input:str):
    text_input = remove_accented_chars(input.lower())
    text_input = expand_contractions(text_input)
    text_input = del_punctuations(text_input)
    text_input_tokenized = nltk.word_tokenize(text_input)
    
    stop_words = stopwords.words('english')
    text_input_tokenized = [item for item in text_input_tokenized if item not in stop_words and item != np.nan]
    return " ".join(text_input_tokenized)

