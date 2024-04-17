import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os     
from datetime import datetime
import json

fail_backgroud = []
detect_error = 0
def get_enahnced_img(img,bg_list):

    background = np.random.choice(bg_list, 1, replace=True)
    bg_img = cv2.imread(background.item())
    try:
        enhanced = enhance_image(img,bg_img)
    except AssertionError:
        global detect_error
        detect_error += 1
        print(f"Error can't find image in background this is the {detect_error}th")
        fail_backgroud.append(background)
        bg_list.remove(background)
        # cv2.imshow('bg_img',bg_img)
        # cv2.waitKey(0)
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
    global save_path
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d")
    if not os.path.exists(f'{save_path}/{current_time}/{file_calss}'):
        os.makedirs(f'{save_path}/{current_time}/{file_calss}')
    if cv2.imwrite(f'{save_path}/{current_time}/{file_calss}/{file}',enhanced):
        print(f'{file} saved')

def make_enhanced_img(img,bg_list,file_calss):
    global total_files
    global enhance_num_per_class
    for index in range(enhance_num_per_class):

        enhanced = get_enahnced_img(img,bg_list)
        if enhanced is None:
            index -= 1
            continue

        now = datetime.now()
        current_time = now.strftime("%Y_%d_%H_%M_%S_%f")
        save_image(enhanced,file_calss,f'{total_files}_{current_time}.jpg')
        index += 1

# Read the image
background_path = './bg_number'
img_path = './number'
save_path = './number_enhanced'
enhance_num_per_class = 6000
bg_list = read_background(background_path)
total_files = None

for root,dir,files in os.walk(img_path):
    for file in files:
        total_files = 0
        extend_name = ('.jpg','.png')
        if file.endswith(extend_name):
            file_calss = root.split('/')[-1]
            root_file = os.path.join(root,file)
            img = cv2.imread(root_file)
            make_enhanced_img(img,bg_list,file_calss)
            total_files += 1
if type(fail_backgroud) == list:
    json_string = json.dumps(fail_backgroud)
    with open('data_enhance_fail.json', 'w') as f:
        json.dump(json_string, f)
elif type(fail_backgroud) == np.ndarray:
    json_string = json.dumps(fail_backgroud.tolist())
    with open('data_enhance_fail.json', 'w') as f:
        json.dump(json_string, f)