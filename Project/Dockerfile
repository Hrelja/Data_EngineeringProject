# Dockerfile

FROM python:3.8

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

COPY . .

#install all our requirements with pip install
RUN pip install -r requirements.txt

EXPOSE 5000


CMD [ "python", "./src/app/app.py" ]