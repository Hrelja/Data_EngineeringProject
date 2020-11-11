import pytest

import preprocessing as prep


def test_remove_accented_chars():
    assert prep.remove_accented_chars("café") == "cafe"

def test_expand_contractions():
    assert prep.expand_contractions("I'm fine, I don't like apple") == "I am fine, I do not like apple"

def test_del_punctuations():
    assert prep.del_punctuations("oh! yes. Me, no, you ?") == "oh yes Me no you "

def test_preprocess():
    assert prep.preprocess("Hello, I'm Fine") == "hello I fine"
