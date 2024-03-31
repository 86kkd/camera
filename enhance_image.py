import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os     

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

# Read the image
background_path = './image_328'
img_path = './image_before'
bg_list = read_background(background_path)

index = 0
for root,dir,files in os.walk(img_path):
    for file in files:
        extend_name = ('.jpg','.png')
        if file.endswith(extend_name):
            file_calss = root.split('/')[-1]
            root_file = os.path.join(root,file)
            img = cv2.imread(root_file)
            enhanced = get_enahnced_img(img,bg_list)
            if enhanced is None:
                continue
            if not os.path.exists(f'./image_after/{file_calss}'):
                os.makedirs(f'./image_after/{file_calss}')
            cv2.imwrite(f'./image_after/{file_calss}/{file}',enhanced)
            # index += 1
            # plt.subplot(4,5,index)
            # plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            # if index == 20:
            #     plt.show()
            #     index = 0