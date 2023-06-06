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
import matplotlib.pyplot as plt

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_segments, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode
import pyttsx3
import speech_recognition as sr
from PIL import Image

tts = pyttsx3.init()
rate = tts.getProperty('rate')
tts.setProperty('rate', rate-40)
volume = tts.getProperty('volume')
tts.setProperty('volume', volume+0.9)
voices = tts.getProperty('voices')
tts.setProperty('voice', 'zh-CN')
for voice in voices:
    if voice.name == 'Anna':
        tts.setProperty('voice', voice.id)


def imgdata():

    # 指定文件夹路径
    folder_path = 'test'

    # 获取文件夹中的所有文件名
    file_names = os.listdir(folder_path)

    # 初始化空列表来保存图片数据
    image_data = []

    # 遍历所有文件名
    for file_name in file_names:
        # 确认文件是图片文件
        if file_name.endswith('.jpg') or file_name.endswith('.JPG') or file_name.endswith('.png'):
            # 拼接文件路径
            file_path = os.path.join(folder_path, file_name)
            # 打开图片并将其转换为 numpy 数组格式
            img = Image.open(file_path)
            img_data = np.asarray(img)
            # 添加到 image_data 列表中
            image_data.append(img_data)
    # plt.imshow(test\test1.jpg)
    # plt.show()
def record_volume():
    end_keyword = "結束"
    r = sr.Recognizer()
    with sr.Microphone(device_index=1) as source:
        talk("請稍等，系統正在進行初始化...")
        r.adjust_for_ambient_noise(source)
        talk("初始化完成，請講話...")
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
                run_video()
                word = "很安全"
                talk(word)
            elif "燈號" in text:
                run_video()
                word = "綠燈"
                talk(word)
            elif text == end_keyword:
                word = "程式已結束"
                talk(word)
                break
            else:
                word = "請再說一次"
                talk(word)
        except sr.UnknownValueError:
            talk("無法辨識")
        except sr.RequestError as e:
            talk(f"無法連接到Google Speech Recognition服務：{e}")
    
    
def talk( word ):
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
# def run_image(img_size=640, stride=32):

#     image_path = './test.JPG'
#     save_path = './'
#     weights = r'./best.pt'
#     device = 'cpu'
#     save_path += os.path.basename(image_path)

#     # 导入模型
#     model = torch.hub.load('C:/Users/liujoe1101/yolov5' , 'custom' , path = 'C:/Users/liujoe1101/yolov5/best.pt' , source='local')
#     img_size = check_img_size(img_size, s=stride)
#     names = model.names

#     # Padded resize
#     img0 = cv2.imread(image_path)
#     img = letterbox(img0, img_size, stride=stride, auto=True)[0]

#     # Convert
#     img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#     img = np.ascontiguousarray(img)

#     img = torch.from_numpy(img).to(device)
#     img = img.float() / 255.0   # 0 - 255 to 0.0 - 1.0
#     img = img[None]     # [h w c] -> [1 h w c]

#     # inference
#     pred = model(img)
#     pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

#     # plot label
#     det = pred[0]
#     annotator = Annotator(img0.copy(), line_width=3, example=str(names))
#     if len(det):
#         det[:, :4] = scale_segments(img.shape[2:], det[:, :4], img0.shape).round()
#         for xyxy, conf, cls in reversed(det):
#             c = int(cls)  # integer class
#             label = f'{names[c]} {conf:.2f}'
#             annotator.box_label(xyxy, label, color=colors(c, True))

#     # write video
#     im0 = annotator.result()
#     cv2.imwrite(save_path, im0)
#     print(f'Inference {image_path} finish, save to {save_path}')

    # for xyxy, conf, cls in reversed(det):
    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
    #     xywh = [round(x) for x in xywh]
    #     xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2],xywh[3]]

    #     cls = names[int(cls)]
    #     conf = float(conf)

def run_video(img_size=640, stride=32,):

    video_path = "./test1.mp4"
    save_path = r"./demo.mp4"
    weights = r'./best.pt'
    device = torch.device('cpu')

    model = torch.hub.load('C:/Users/liujoe1101/yolov5' , 'custom' , path = 'C:/Users/liujoe1101/best.pt' , source='local')
    img_size = check_img_size(img_size, s=stride)
    names = model.names
    cap = cv2.VideoCapture(video_path)
    frame = 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save_path += os.path.basename(video_path)
    #vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while frame < frames:
        ret_val, img0 = cap.read()
        if not ret_val:
            break
        frame += 1
        #print(f'video {frame}/{frames} {save_path}')

        # Padded resize
        img = letterbox(img0, img_size, stride=stride, auto=True)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to('cpu')
        img = img.float() / 255.0
        img = img[None]

        # inference
        pred1 = model(img)
        pred2 = non_max_suppression(pred1, conf_thres=0.25, iou_thres=0.45, max_det=1000)

        # plot label
        det = pred2[0]
        annotator = Annotator(img0.copy(), line_width=3, example=str(names))
        if len(det):
            det[:, :4] = scale_segments(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

        im0 = annotator.result()
        #vid_writer.write(im0)

        for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                xywh = [round(x) for x in xywh]
                xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2],xywh[3]]

                cls = names[int(cls)]
                conf = float(conf)

        
            
        
        cv2.namedWindow('image', 0)
        cv2.resizeWindow('image', 600,600)
        cv2.imshow('image' , im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    #vid_writer.release()
    cap.release()
    print(f'{video_path} finish, save to {save_path}')




if __name__ == '__main__':
    # imgdata()
    record_volume()
    # run_image()
