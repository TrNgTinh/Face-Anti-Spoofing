import cv2
from PIL import Image
from test import test
import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_F1.jpg",
        help="image used to test")
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    count = 0 
    while cap.isOpened():
        isSuccess,frame = cap.read()
        if isSuccess:
            test(frame,args.model_dir, args.device_id) 
            cv2.imshow('face Capture', frame)
        count += 1
        #if count ==1:
        #    break
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()    