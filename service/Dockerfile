FROM python:3.8.1-slim-buster

RUN pip3 install --upgrade pip
COPY ["requirements.txt", "requirements.txt"]
RUN pip3 install -r /requirements.txt

RUN mkdir app
ADD . /app
WORKDIR /app

EXPOSE 8082

CMD ["python3", "/app/server.py", "8082"]
