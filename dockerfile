FROM python:3.10.11
WORKDIR /
ADD . /project
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y libgl1-mesa-glx
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY yolov5 /home
ENTRYPOINT FLASK_APP=/home/server.py flask run --host=0.0.0.0