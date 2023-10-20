# syntax=docker/dockerfile:1

FROM python:3.7.9
RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /flask/

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python3" , "./app.py"]
