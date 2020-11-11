import requests
from requests.api import request
import app


def test_index():
    responce = requests.get('http://0.0.0.0:5000')
    assert responce.status_code == 200

def test_analyze_sentiment():
    responce = requests.get('http://0.0.0.0:5000')
    assert responce.status_code == 200
    assert app.analyze_sentiment("good") == 1
