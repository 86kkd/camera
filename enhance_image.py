import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os     
from datetime import datetime
import json
import argparse
from crope_image import crope_image

parser = argparse.ArgumentParser(description='crope image or enhace image')
parser.add_argument('--crope',action='store_true',
                    help='corpe image or enhance')
parser.add_argument('--save-path',default='./image_enhanced',metavar='str',
                    help='path to save enhanced image')
parser.add_argument('--per-enhance',default=30,metavar='int',
                    help='num enhanced image per image in each class')
parser.add_argument('--bg-path',default='./bg_img',metavar='str',
                    help='path to bg_img if crope selected it is the path to corping img')
parser.add_argument('--img-path',default='./image',metavar='str',
                    help='path to img_path if crope selected it is ignored')
args = parser.parse_args()
fail_backgroud = np.array([])

def gamma_transform(img, gamma):
    is_gray = img.ndim == 2 or img.shape[1] == 1
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    illum = hsv[..., 2] / 255.
    illum = np.power(illum, gamma)
    v = illum * 255.
    v[v > 255] = 255
    v[v < 0] = 0
    hsv[..., 2] = v.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def get_enahnced_img(img,bg_list):
    global fail_backgroud
    background = np.random.choice(bg_list, 1, replace=True)
    bg_img = cv2.imread(background.item())
    try:
        if args.crope:
            enhanced = crope_image(img)
        else:
            enhanced = enhance_image(img,bg_img)
            enhanced = gamma_transform(enhanced,np.random.uniform(0.3,3,[1]))
    except AssertionError:
        print(f"Error can't find image in background this is the {len(fail_backgroud)}th")
        fail_backgroud = np.append(fail_backgroud,background)
        bg_list.remove(background)
        if args.crope:
            return None
        else:
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
    global args
    save_path = args.save_path
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d")
    if not os.path.exists(f'{save_path}/{current_time}/{file_calss}'):
        os.makedirs(f'{save_path}/{current_time}/{file_calss}')
    if cv2.imwrite(f'{save_path}/{current_time}/{file_calss}/{file}',enhanced):
        print(f'{file} saved')

def make_enhanced_img(img,bg_list,file_calss):
    global args
    if args.crope:
        enhance_num_per_class = 1
    else:
        enhance_num_per_class = args.per_enhance

    for index in range(enhance_num_per_class):

        enhanced = get_enahnced_img(img,bg_list)
        if enhanced is None:
            index -= 1
            continue

        now = datetime.now()
        current_time = now.strftime("%Y_%d_%H_%M_%S_%f")
        save_image(enhanced,file_calss,f'{current_time}.jpg')
        index += 1

def split_files(files):
    for file in files:
        extend_name = ('.jpg','.png')
        if file.endswith(extend_name):
            file_calss = root.split('/')[-1]
            root_file = os.path.join(root,file)
            img = cv2.imread(root_file)
            make_enhanced_img(img,bg_list,file_calss)

if __name__ == "__main__":
    # Read the image
    background_path = args.bg_path
    if args.crope:
        img_path = args.bg_path 
    else:
        img_path = args.img_path
    
    bg_list = read_background(background_path)

    try:
        for root,dir,files in os.walk(img_path):
            split_files(files)
    except KeyboardInterrupt:
        pass


    json_string = json.dumps(fail_backgroud.tolist())
    with open('data_enhance_fail.json', 'w') as f:
        json.dump(json_string, f,indent=4)
    print('\n\033[93mFinished enahcenment fail bg image in data_enhance_fail.json\033[0m')