from flask import Flask, request, render_template
from redis import Redis, RedisError, StrictRedis
import json

app = Flask(__name__)

def add_sentence(sid, sentence):
    sentence = {'sentence':sentence}
    status = "success"
    try:
        redis_client.set(sid, json.dumps(sentence))
    except RedisError:
        status = 'fail'
    return status
    
@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		details = request.form
		if details['form_type'] == 'add_user':
			return add_sentence(details['sid'], details['sentence'])
	return render_template('index.html')

if __name__ == '__main__':
	redis_client = StrictRedis(host='redis', port=6379)
	app.run(host='0.0.0.0')