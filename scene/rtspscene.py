import cv2
import time
import datetime
import threading
import numpy as np
from scene import Log
lock = threading.Lock()
# classes = ["electric", "bicycle"]
import os
WINDOW_NAME = 'CameraDemo'
class RtspScene:
    def __init__(self, cameraId, rtsp, chedou_model, scenename, points):
        self.rtspid = cameraId
        self.rtsp = rtsp
        self.chedou_model = chedou_model
        self.name = scenename
        self.cap = self.open_cam_rtsp(rtsp, 1920, 1080, 25)
        # self.cap = cv2.VideoCapture(rtsp)
        self.points = points
        self.image = None
        self.flag = 1
        self.closeflag = 1
        self.lastNum = 0
        self.maxNum = 3
        self.imgList = []
        self.id = 0
        self.turnRatio = 0.001
        self.ifrun = False

    def osrun(self, direction, delayTime):
        #0 停止
        #1 左
        #2 右
        #4 上
        #8 下
        print(direction,delayTime)
        #if delayTime>0.5:
        #    delayTime = 0.5
        #id delayTime < 0.125
        delayTime = 0.15
        self.ifrun = True
        os.system("/home/chase/uartapp /dev/ttyUSB0 0 "+str(direction))
        time.sleep(0.3)
        if delayTime<0.15:
            time.sleep(0.1)
        else:
            time.sleep(0.1)
        os.system("/home/chase/uartapp /dev/ttyUSB0 0 0")
        self.ifrun = False
    def open_window(self, width, height):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, width, height)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson TX2/TX1')

    def start(self):
        t = threading.Thread(target=self.readFrame)
        t.setDaemon(True)
        t.start()
        self.open_window(1920, 1080)
        self.id = 1
        while(True):
            #time.sleep(1)
            #if self.ifrun:
            #  time.sleep(0.01)
            #   continue
            if self.image is not None:
                ori_im = self.image.copy()
                #ori_im = cv2.imread('/home/chase/ddd/1618471567.1641395.jpg')
                try:
                    #检测
                    start = time.time()
                    chedou_bboxes, chedou_labels = self.chedou_model.detect(ori_im, self.points)
                    print(time.time()-start)
                    #print(chedou_bboxes)
                    #判断
                    x1 =ori_im.shape[1]/5*2
                    y1 =ori_im.shape[0]/5*2-50
                    x2 = ori_im.shape[1]/5*3
                    y2 = ori_im.shape[0]/5*3+50
                    cv2.rectangle(ori_im, (int(x1), int(y1)),(int(x2), int(y2)), (0,0,255), 5)
                    if len(chedou_bboxes) > 0:
                        maxId = np.argmax(chedou_bboxes, axis=0)[-1]
                        bbox = chedou_bboxes[maxId]
                        label = chedou_labels[maxId]
                        center = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
                        print(bbox)
                        cv2.rectangle(ori_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 5)
                        if not self.ifrun:
                            
                            if center[0] > ori_im.shape[1]/5*3:
                                print('向右转')
                                delayTime=(center[0]-ori_im.shape[1]/2)*self.turnRatio
                                t = threading.Thread(target=self.osrun, args=(1, delayTime))
                                t.start()
                                #self.osrun(direction=1, delayTime=(center[0]-ori_im.shape[1]/2)*self.turnRatio)
                                #向右转
                            elif center[0] < ori_im.shape[1]/5*2:
                                print('向左转')
                                delayTime=(-center[0]+ori_im.shape[1]/2)*self.turnRatio
                                t = threading.Thread(target=self.osrun, args=(2, delayTime))
                                t.start()
                                #self.osrun(direction=2, delayTime=(-center[0]+ori_im.shape[1]/2)*self.turnRatio)
                                #向左转
                           
                            elif center[1] > ori_im.shape[0]/5*3+50:
                                print('向up转')
                                delayTime=(-center[1]+ori_im.shape[1]/2)*self.turnRatio
                                t = threading.Thread(target=self.osrun, args=(8, delayTime))
                                t.start()
                                #self.osrun(direction=2, delayTime=(-center[0]+ori_im.shape[1]/2)*self.turnRatio)
                                #向up转
                            elif center[1] < ori_im.shape[0]/5*2-50:
                                print('向down转')
                                delayTime=(-center[0]+ori_im.shape[0]/2)*self.turnRatio
                                t = threading.Thread(target=self.osrun, args=(4, delayTime))
                                t.start()
                                #self.osrun(direction=2, delayTime=(-center[0]+ori_im.shape[1]/2)*self.turnRatio)
                                #向up转 
                        

                    if self.id % 10 == 0:
                        cv2.imshow(WINDOW_NAME,ori_im)
                        cv2.waitKey(10)
                        cv2.imwrite('/home/chase/IIII/'+str(time.time())+'.jpg', self.image)
                    self.id += 1
                    if self.id > 100:
                        self.id = 1
                    end = time.time()
                    print(str(self.rtspid)+':'+"time: {}s, fps: {}".format(end - start, 1 / (end - start)))
                    #cv2.imwrite('/home/chase/test.jpg', ori_im)
                except Exception as e:
                    Log.logger.exception(e)
                    Log.logger.error(str(self.rtspid)+'模型错误')
                    break
        Log.logger.warning(str(self.rtspid)+'场景关闭，图片释放')


    # thread: read frame
    def readFrame(self):
        f = 0
        time.sleep(2)
        Log.logger.info('startrtsp')
        while (True):
            try:
                ret, frame = self.cap.read()  # 获取一帧
                if ret:
                    self.image = frame
                    f = 0
                else:
                    Log.logger.warning(str(self.rtspid) + '无法获取画面')
                    self.cap.release()
                    self.image = None
                    if f < 10:
                        time.sleep(15)
                        self.cap = cv2.VideoCapture(self.rtsp)
                        Log.logger.warning('尝试重联')
                        f += 1
                    else:
                        break
            except Exception as e:
                Log.logger.exception(e)
        self.cap.release()
        Log.logger.warning('cap.release()')
    def open_cam_rtsp(self, uri, width, height, latency):
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! omxh264dec ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(uri, latency, width, height)
        print(gst_str)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
