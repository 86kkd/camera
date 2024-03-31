import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os     

fail_backgroud = []
def get_enahnced_img(img,bg_list):

    for background in np.random.choice(bg_list, 10, replace=False):
        bg_img = cv2.imread(background)
        try:
            enhanced = enhance_image(img,bg_img)
        except :
            print("Error can't find image in background")
            fail_backgroud.append(background)
            # TODO check if deleted
            del background
            return None
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

for root,dir,files in os.walk(img_path):
    for file in files:
        extend_name = ('.jpg','.png')
        if file.endswith(extend_name):
            file_calss = root.split('/')[-1]
            file = os.path.join(root,file)
            img = cv2.imread(file)
            enhanced = get_enahnced_img(img,bg_list)
            if enhanced is None:
                continue
            plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            plt.show()