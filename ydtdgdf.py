import argparse
import os
import platform
import sys
from pathlib import Path
import torch
import numpy as np
import time
import pyttsx3
import speech_recognition as sr
import time
import threading
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode
import pyttsx3
import speech_recognition as sr
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

def talk(text):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate', 180)
        if not engine._inLoop:
            engine.startLoop(False)
        engine.say(text)
        engine.iterate()
        engine.endLoop()
        engine.runAndWait()
    except Exception as e:
        print(f"Error: {e}")


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    current_det_signal = pyqtSignal(list)

    def __init__(self):
        QThread.__init__(self)
        self.video_path = "./test_night.mp4"
        self.device = 'cuda'

        self.model = torch.hub.load('C:/Users/liujoe1101/yolov5', 'custom',
                                    path='C:/Users/liujoe1101/yolov5/123.pt', source='local')
        self.img_size = check_img_size(640, s=32)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        self.cap = cv2.VideoCapture(self.video_path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame = 0

    def run(self):
        while self.frame < self.frames:
            ret_val, img0 = self.cap.read()
            if not ret_val:
                break
            self.frame += 1

            # Padded resize
            img = letterbox(img0, self.img_size, stride=32, auto=True)[0]

            # Convert
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0
            img = img[None]

            # inference
            pred1 = self.model(img)
            pred2 = non_max_suppression(pred1, conf_thres=0.25, iou_thres=0.45, max_det=1000)

            # plot label
            det = pred2[0]

            annotator = Annotator(img0.copy(), line_width=3, example=str(self.names))
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                xywh = [round(x) for x in xywh]
                xywh = [xywh[0], xywh[1], xywh[2], xywh[3]]
                c = int(cls)
                x = xywh[0]
                if (x >= 759 and x <= 1059 and c == 4):
                    d_car = float((3130.8*140)/xywh[3]/100)
                    if(d_car <= 9.5):
                        talk('行徑路線有車輛阻擋請注意')


            im0 = annotator.result()
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            self.change_pixmap_signal.emit(im0)
            self.current_det_signal.emit(det.tolist())

        self.cap.release()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.video_path = "./test.mp4"

        self.thread = QThread()
        self.worker = VideoThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.change_pixmap_signal.connect(self.update_image)
        self.worker.current_det_signal.connect(MainWindow.update_current_det)
        self.thread.start()

    def initUI(self):
        self.setWindowTitle("Detection")
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 1360, 768)
        self.show()

    def update_image(self, img):
        h, w, ch = img.shape
        bytesPerLine = ch * w
        qImg = QImage(img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(pixmap)

    def get_current_det(cls):
        return cls.det
    
    def update_current_det(det):
        MainWindow.det = det
    
    def run_gui():
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        app.exec_()

def check_results1(data):
    for *xyxy, conf, cls in reversed(data):
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
        xywh = [round(x) for x in xywh]
        xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2], xywh[3]]
        x = xywh[0]
        c = int(cls)
        conf = float(conf)
        if ((x >= 581 and x<= 1190)and (c == 1 or c ==3)):
            f=1
            break
        if ((x >= 581 and x<= 1190)and (c == 0 or c ==2)):
            f=2
            break
        else:
            f=0
    if (f==1):
        talk('前方綠燈')
    elif( f==2 ):
        talk('前方紅燈')
    else:
        talk('前方沒有燈號')
    f=0
            
    
def check_results2(data):
    for *xyxy, conf, cls in reversed(data):
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
        xywh = [round(x) for x in xywh]
        xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2], xywh[3]]
        x = xywh[0]
        c = int(cls)
        conf = float(conf)
        if ((x >= 581 and x<= 1190)and (c == 4 or c ==5)):
            f=1
            break
        else:
            f=0
    if (f==1):
        talk('前方有車')
    else:
        talk('前方沒車')
    f=0
      

def check_results3(data):
    for *xyxy, conf, cls in reversed(data):
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
        xywh = [round(x) for x in xywh]
        xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2], xywh[3]]
        x = xywh[0]
        c = int(cls)
        conf = float(conf)
        if ((x >= 581 and x<= 1190)and c == 6 ):
            f=1
            break
        else:
            f=0
    if (f==1):
        talk('前方有人')
    else:
        talk('前方沒人')
    f=0


if __name__ == '__main__':
    end_keyword = "結束"
    r = sr.Recognizer()
    with sr.Microphone(device_index=1) as source:
        talk("請稍等，系統正在進行初始化...")
        r.adjust_for_ambient_noise(source)
        talk("初始化完成")
        gui_thread = threading.Thread(target=MainWindow.run_gui)
        gui_thread.start()
    while True:
        with sr.Microphone(device_index=1) as source:
            talk("正在辨識...")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source)
        try:
            query = r.recognize_google(audio, language='zh-CN')
            text = query.lower()
            talk(f'您說：{query}')
            if "路況" in text:
                det = MainWindow.get_current_det(MainWindow)
                check_results2(det)
            elif "燈號" in text:
                det = MainWindow.get_current_det(MainWindow)
                check_results1(det)
            elif "人" in text:
                det = MainWindow.get_current_det(MainWindow)
                check_results3(det)
            elif text == end_keyword:
                word = "程式已結束"
                talk(word)
                break
            else:
                talk('再說一次')
        except sr.UnknownValueError:
            talk("無法辨識")
        except sr.RequestError as e:
            talk(f"無法連接到Google Speech Recognition服務：{e}")

