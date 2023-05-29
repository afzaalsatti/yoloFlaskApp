# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import time
import torch
import psutil
#import GPUtil
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov8n.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    cap = cv2.VideoCapture(source)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / vid_stride)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_duration = round(frames / fps)
    objects_detected = {}
    frames_count = 0
    processing_time = 0
    input_fps = 0
    detailed_output =[]
    accuracy = []
    # Initialize variables for tracking CPU, memory, and GPU usage
    cpu_usage_list = []
    mem_usage_list = []
    gpu_usage_list = []
    source = str(source)
    from ultralytics import YOLO
    cpu_usage = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    mem_usage = mem.used / mem.total * 100
    start_time = time.time()
    cpu_usage_list.append(cpu_usage)
    mem_usage_list.append(mem_usage)
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
    axis=[]
    names = model.names
    # Perform object detection on an image using the model
    results = model(source)
    cpu_usage_list.append(cpu_usage)
    mem_usage_list.append(mem_usage)
    for result in results:
        processing_time += result.speed["inference"]
        detected_Objects={}
        frames_count+=1;
        frame_info = {"frame_number":frames_count, "detected_Objects":{} }
        axis = []
        identified_frame_object = [];
        for box in result.boxes:
            accuracy.append(box.conf.item());
            class_name =names[int(box.cls)]
            if class_name not in identified_frame_object:
                if class_name in objects_detected:
                    objects_detected[class_name] += 1  # Update value for 'key1'
                    identified_frame_object.append(class_name)
                else:
                    objects_detected[class_name] = 1  # Insert new object for 'key1'
                    identified_frame_object.append(class_name)
            axis.append({class_name: (xyxy2xywh(torch.tensor(box.xyxy).view(1, 4))).view(-1).tolist(), })
        frame_info["detected_Objects"] = objects_detected
        frame_info["axis"] = axis
        detailed_output.append(frame_info)
    end_time = time.time()
    for detected_object_class in objects_detected.keys():
        frame_visibility_duration = int(objects_detected.get(detected_object_class))/fps
        objects_detected[detected_object_class] = frame_visibility_duration
    cpu_usage = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    mem_usage = mem.used / mem.total * 100
    import platform
    uname = platform.uname()
    # Calculate elapsed time and output frame rate
    elapsed_time = end_time - start_time
    frame_rate = frames_count / elapsed_time
    # Calculate average usage
    avg_cpu_usage = sum(cpu_usage_list) / len(cpu_usage_list)
    avg_mem_usage = sum(mem_usage_list) / len(mem_usage_list)

    server_response = {}
    print("-------------------Strating YoloV8 log--------------------------")
    print("-------------------System Information--------------------------")
    print(f"System: {uname.system}")
    print(f"Release: {uname.release}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")
    server_response["system_info"] = {"system":uname.system, "release":uname.release,"machine":uname.machine}
    
    print("-------------------Resource Usage--------------------------")

    # Print average usage
    print(f'Average CPU usage: {avg_cpu_usage:.2f}%')
    print(f'Average memory usage: {avg_mem_usage:.2f}%')
    server_response["resource_usage"] = {"cpu_usage":avg_mem_usage, "memory":avg_mem_usage}
    #for i, usage in enumerate(avg_gpu_usage):
        #print(f'Average GPU {i+1} usage: {usage:.2f}%')

    print("-------------------Time Consumption--------------------------")
    print(f'Start Time: {start_time:.2f}s')
    print(f'End Time: {end_time:.2f}s')
    print(f'Elapsed time: {elapsed_time:.2f} s')
    processing_time = processing_time/1000
    print(f'Total Processing Time:{processing_time:.2f}s')
    server_response["time_consumption"] = {"start_time":start_time, "end_time":end_time
                                          , "elapsed_time":elapsed_time, "processing_time":processing_time}

    print("-------------------Frame Rate--------------------------")
    print("frames count:",frames_count)
    print(f'Input Frame rate: {fps:.2f} fps')
    print(f'Output Frame rate: {frame_rate:.2f} fps')
    server_response["frames_info"] = {"frames_count":frames_count,"input_frame_rate":fps, "output_frame_rate":frame_rate}

    print("-------------------Object Detection--------------------------")
    print("detections:",objects_detected)
    average_conf = sum(accuracy)/len(accuracy)
    print(f'Accuracy conf: {average_conf:.2f} %')
    print("Detailed Output For Graphs",detailed_output)
    print("-------------------End YoloV8 log--------------------------")
    server_response["object_detection"] = {"detections":objects_detected, "average_conf":average_conf, "detailed_output":detailed_output}
    return server_response

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
