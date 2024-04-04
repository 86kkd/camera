import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os     
from datetime import datetime


fail_backgroud = []
detect_error = 0
def get_enahnced_img(img,bg_list):

    background = np.random.choice(bg_list, 1, replace=True)
    bg_img = cv2.imread(background.item())
    try:
        enhanced = enhance_image(img,bg_img)
    except :
        global detect_error
        detect_error += 1
        print(f"Error can't find image in background this is the {detect_error}th")
        fail_backgroud.append(background)
        bg_list.remove(background)
        return get_enahnced_img(img,bg_list)
    return enhanced

def read_background(path):
    sample_list = []
    
    for root,dir,files in os.walk(path):
        for file in files :
            extend_name = ('.jpg','.png')
            if file.endswith(extend_name):
                sample_list.append(os.path.join(root,file))

    return sample_list

def save_image(enhanced,file_calss,file):
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d")
    if not os.path.exists(f'./image_after{current_time}/{file_calss}'):
        os.makedirs(f'./image_after{current_time}/{file_calss}')
    if cv2.imwrite(f'./image_after{current_time}/{file_calss}/{file}',enhanced):
        print(f'{file} saved')

def make_enhanced_img(img,bg_list,file_calss):
    index = 0
    while(index<30):
        enhanced = get_enahnced_img(img,bg_list)
        if enhanced is None:
            index -= 1
            continue

        now = datetime.now()
        current_time = now.strftime("%Y_Y%d_%H_H%M_M%S_%f")
        save_image(enhanced,file_calss,f'{index}_{current_time}.jpg')
        index += 1
# Read the image
background_path = './image/image42_100'
img_path = './image/image_before'
bg_list = read_background(background_path)

for root,dir,files in os.walk(img_path):
    for file in files:
        extend_name = ('.jpg','.png')
        if file.endswith(extend_name):
            file_calss = root.split('/')[-1]
            root_file = os.path.join(root,file)
            img = cv2.imread(root_file)
            make_enhanced_img(img,bg_list,file_calss)