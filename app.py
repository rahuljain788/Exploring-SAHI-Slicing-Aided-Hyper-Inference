from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.prediction import visualize_object_predictions
from numpy import asarray
import matplotlib.pyplot as plt
import cv2

yolov8_model_path = 'models/yolov8s.pt'
download_yolov8s_model(destination_path=yolov8_model_path)

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.3,
    device="cpu"
)

# Normal prediction
result = get_prediction('input/traffic-scaled.jpg', detection_model)
result.export_visuals(export_dir='output/')

# SAHI Sliced Prediction
result = get_sliced_prediction(
    "input/traffic-scaled.jpg",
    detection_model,
    slice_height = 512,
    slice_width = 512,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

img = cv2.imread("input/traffic-scaled.jpg", cv2.IMREAD_UNCHANGED)
img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
numpydata = asarray(img_converted)
visualize_object_predictions(
    numpydata,
    object_prediction_list = result.object_prediction_list,
    hide_labels = 1,
    output_dir='output',
    file_name = 'result',
    export_format = 'png'
)

image = cv2.imread('output/result.png')

scale_percent = 40  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("output", resized)
cv2.waitKey(0)