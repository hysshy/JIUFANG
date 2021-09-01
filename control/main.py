# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from scene.rtspscene import RtspScene
import threading
#import gi
#gi.require_version('Gtk', '3.0')
#from module.mmdetection.mmhy import mmhy_detect
from module.yolov5 import yolo_model


def startRtspScene(cameraId, rtsp, model, sceneName, points):
    sceneItem = RtspScene(cameraId,rtsp,model,sceneName, points)
    sceneItem.start()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rtspList = []
    yolo_model.init_model('0')
    startRtspScene(None, 'rtsp://admin:windaka123@10.10.5.245', yolo_model, None, None)
    # startRtspScene(None, 'rtsp://admin:admin123@10.10.5.242:554', yolo_model, None, None)
    # for rtspItem in rtspList:
    #     if rtspItem[1] == 'rtsp':
    #         t = threading.Thread(target=startRtspScene, args=(rtspItem[0], rtspItem[2], mmhy_detect, rtspItem[1], rtspItem[3]))
    #         t.start()
    # t.join()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
