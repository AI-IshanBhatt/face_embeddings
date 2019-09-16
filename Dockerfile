FROM python:3.6.9-slim

MAINTAINER Ishan Bhatt <ishan_bhatt@hotmail.com>

RUN mkdir FACE_RECOGNITION

COPY requirements.txt FACE_RECOGNITION/requirements.txt

RUN apt-get update && apt-get install -y build-essential \
	cmake

RUN pip install -r FACE_RECOGNITION/requirements.txt

COPY . FACE_RECOGNITION

WORKDIR FACE_RECOGNITION

ENTRYPOINT ["python", "face_embedding/face_embedding.py"]
