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
import threading
import matplotlib.pyplot as plt
import pylab
import imageio

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_segments, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode
import pyttsx3
import speech_recognition as sr
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
import numpy as np  
import cv2  
from moviepy.editor import *
from datetime import datetime, timedelta

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = datetime.now()
        self.end_time = None

    def stop(self):
        self.end_time = datetime.now()

    def elapsed_time(self):
        if self.start_time is None:
            return 0
        elif self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds()
        else:
            return (self.end_time - self.start_time).total_seconds()


tts = pyttsx3.init()
rate = tts.getProperty('rate')
tts.setProperty('rate', rate - 40)
volume = tts.getProperty('volume')
tts.setProperty('volume', volume + 0.9)
voices = tts.getProperty('voices')
tts.setProperty('voice', 'zh-CN')
for voice in voices:
    if voice.name == 'Anna':
        tts.setProperty('voice', voice.id)


def talk(word):
    tts.say(word)
    tts.runAndWait()


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


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.det = self.run_image()

    def initUI(self):
        self.setWindowTitle('TEST')

        # 用于显示图像的 QLabel
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        # 将 QLabel 添加到 QVBoxLayout 中
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        # 设置 QWidget 的 layout
        self.setLayout(layout)

    def run_image(self, img_size=640, stride=32,):

        image_path = './frame1.jpg'
        weights = r'./best_second.pt'
        device = 'cpu'

        # 导入模型
        model = torch.hub.load('C:/Users/liujoe1101/yolov5', 'custom', path='C:/Users/liujoe1101/yolov5/best_second.pt', source='local')
        img_size = check_img_size(img_size, s=stride)
        names = model.names

        # Padded resize
        img0 = cv2.imread(image_path)
        img = letterbox(img0, img_size, stride=stride, auto=True)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0   # 0 - 255 to 0.0 - 1.0
        img = img[None]     # [h w c] -> [1 h w c]

        # inference
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

        # plot label
        det = pred[0]
        annotator = Annotator(img0.copy(), line_width=3, example=str(names))
        if len(det):
            det[:, :4] =  scale_segments(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

        im0 = annotator.result()

        height, width, channel = im0.shape
        bytesPerLine = 3 * width
        qImg = QPixmap(QImage(im0.data, width, height, bytesPerLine, QImage.Format_RGB888))
        self.image_label.setPixmap(qImg)
        self.resize(width, height)
        return(det)




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
def orytb():
    v=1
    end_keyword = "結束"
    r = sr.Recognizer()
    with sr.Microphone(device_index=1) as source:
        talk("請稍等，系統正在進行初始化...")
        r.adjust_for_ambient_noise(source)
        talk("初始化完成")
        
        window = MainWindow()
        window.show()
        det = window.det
        data = det
    while (v==1):
        with sr.Microphone(device_index=1) as source:
            talk("正在辨識...")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, phrase_time_limit=3)
        try:
            query = r.recognize_google(audio, language='zh-CN')
            text = query.lower()
            talk(f'您說：{query}')
            if "路況" in text:
                check_results2(data)
            elif "燈號" in text:
                check_results1(data)
            elif "人" in text:
                check_results3(data)
            elif  end_keyword in text:
                word = "程式已結束"
                talk(word)
                v=0
                
            else:
                word = "沒有關鍵字"
                talk(word)
        except sr.UnknownValueError:
            talk("無法辨識")
        except sr.RequestError as e:
            talk(f"無法連接到Google Speech Recognition服務：{e}")
if __name__ == '__main__':
    cap = cv2.VideoCapture('test.mp4')  
    video = VideoFileClip("test.mp4")
    timer = Timer() 
    timer.start()
    t=0
    app = QApplication(sys.argv)
    while(cap.isOpened()):  
        ret, frame = cap.read()  
        cv2.imshow('image', frame) 
        k = cv2.waitKey(20)  
        #q键退出
        if (k & 0xff == ord('q')):  
           break 
        if(k & 0xff == ord('g')):
            timer.stop()
            t=timer.elapsed_time()+t
            print(t)
            frame = video.save_frame("frame1.jpg", t )
            orytb()
            timer.start()
            

        

cap.release()  
cv2.destroyAllWindows()
