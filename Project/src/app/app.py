from flask import Flask, request, render_template
import json
import joblib


app = Flask(__name__)

pipeline = joblib.load('src/model/pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')
    
def analyze_sentiment(sentence):
    return pipeline.predict([sentence])[0]

    
@app.route('/', methods=['POST'])
def index():
    details = request.form
    sentence = details['sentence']
    pred = analyze_sentiment(sentence)
    dico = {'1':"positive", '-1':'negative', '0':'neutral' }
    sentiment = dico[str(pred)]
    return render_template('index.html', prediction_response='prediction is {}'.format(sentiment))

if __name__ == '__main__':
	app.run(host='0.0.0.0')
