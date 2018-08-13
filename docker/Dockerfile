from python:3

RUN apt update && apt-get install -y postgresql

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt -e git+https://github.com/heschao/craniutil/#egg=craniutil


RUN adduser --disabled-password --gecos '' camper
USER camper