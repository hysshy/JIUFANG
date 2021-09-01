import os
import sys
sys.path.insert(0, '/home/chase/yolov5-master')
from utils.torch_utils import select_device, load_classifier, time_synchronized
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
import cv2
import torch
import numpy as np
from utils.datasets import letterbox
# weights = '/home/chase/yolov5-master/tests/exp10/weights/best.pt'
weights = '/home/chase/Downloads/last.pt'
img_size = 640
conf_thres =0.25
iou_thres = 0.35

def init_model(gpuId):
    global device
    device = select_device(gpuId)
    global model
    model = attempt_load(weights, map_location=device)
    global stride
    stride = int(model.stride.max())  # model stride
    # imgsz = check_img_size(imgsz0, s=stride)  # check img_size
    model.half()  # to FP16


def detect(img, points=None):
    if isinstance(img, str):
        img = cv2.imread(img)
    img0 = img.copy()

    # Padded resize
    img = letterbox(img, img_size, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=False)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    det = pred[0]
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

    # Print results
    for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class

    # Write results
    # det = det.tolist()
    bboxes = det[:,0:5].tolist()
    labels = det[:,-1].tolist()
    return np.array(bboxes), np.array(labels)

if __name__ == '__main__':
    init_model('0')
    img = cv2.imread('/home/chase/aaaa/1627952536.7493541.jpg')
    bboxes, labels = detect(img)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    cv2.imwrite('/home/chase/busdraw.jpg', img)

