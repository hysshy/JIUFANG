# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
cv2.namedWindow('WINDOW_NAME')
def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
           'rtph264depay ! h264parse ! omxh264dec ! '
           'nvvidconv ! '
           'video/x-raw, width=(int){}, height=(int){}, '
           'format=(string)BGRx ! '
           'videoconvert ! appsink').format(uri, latency, width, height)
    print(gst_str)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
cap = open_cam_rtsp('rtsp://admin:suizhi123@10.10.5.245:554', 1920, 1080, 25)
while 1:
    ret, frame = cap.read()
    cv2.imshow('WINDOW_NAME', frame)
    cv2.waitKey(1)
    print(ret)
