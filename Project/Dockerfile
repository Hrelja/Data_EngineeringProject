# Dockerfile

FROM python:3.8

#copy current folder . into our container folder /app
COPY . /app

WORKDIR /app 

#install all our requirements with pip install
RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]