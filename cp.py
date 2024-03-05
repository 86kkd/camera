import os 

path = 'enhance_img'

for root,dirs,files in os.walk(path):
    print(dirs)
    for root,dirs,files in dirs[i]: