from flask import Flask, render_template, request
import os
#import helper methods
from detectYolov8 import run as yolov8
from detect import run as yolov5

app = Flask(__name__)
@app.route("/")
def index_page():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file uploaded.', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'Empty filename.', 400
    # Specify the folder name you want to create
    folder_name = 'temp_uploads'

    # Get the current working directory
    current_directory = os.getcwd()

    # Create the folder path by joining the current directory and folder name
    folder_path = os.path.join(current_directory, folder_name)
    # Create the folder
    os.makedirs(folder_path, exist_ok=True)

    # Save the uploaded file to the specified folder
    file_path = os.path.join(folder_path, file.filename)
    file.save(file_path)
    yolov8_response =  yolov8("yolov8n.pt", file_path )
    yolov5_response =  yolov5("yolov5s.pt", file_path )
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"yolov5_response":yolov5_response, "yolov8_response":yolov8_response}
@app.route("/yolov8")
def hello_world():
    return yoloV8("yolov8n.pt", "./data/videos/video3.mp4" )